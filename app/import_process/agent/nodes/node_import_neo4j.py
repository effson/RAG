# 导入基础库：系统、JSON解析、类型注解、多线程
import sys
import json
import re
import os
import threading
from typing import Dict, List, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Neo4j配置：获取数据库名等配置
from app.conf.neo4j_conf import get_config as get_neo4j_config
# 3. Neo4j工具：获取单例Neo4j驱动，实现连接复用
from app.clients.neo4j_utils import get_neo4j_driver
# 4. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 5. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 6. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger, node_log, step_log
# 7. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt
from pymilvus import DataType
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config


# ---- 配置文件 ----

# 允许的关系类型白名单（与 entity_extraction.prompt 中定义一致）
ALLOWED_REL_TYPES = {
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
}

# 允许的实体标签白名单（与 entity_extraction.prompt 中定义一致）
ALLOWED_ENTITY_LABELS = {
    "DEVICE", "PART", "OPERATION", "STEP", "WARNING", "CONDITION", "TOOL",
}

# BGE-M3 Embedding 模型互斥锁——本地 PyTorch 模型多线程共享同一份权重，
# 并发推理会导致内部状态冲突，所有 embedding 调用必须持有此锁
_embedding_lock = threading.Lock()


# ---- 数据结构定义 ----

@dataclass
class EntityRef:
    """去重用的实体引用，以(name, label)为唯一标识"""
    name: str
    label: str
    description: str = ""

    def __hash__(self):
        return hash((self.name.strip(), self.label.strip()))

    def __eq__(self, other):
        if not isinstance(other, EntityRef):
            return False
        return self.name.strip() == other.name.strip() and self.label.strip() == other.label.strip()


@dataclass
class RelationRef:
    """去重用的关系引用，以(head, tail, rel_type)为唯一标识"""
    head: EntityRef
    tail: EntityRef
    rel_type: str

    def __hash__(self):
        return hash((self.head, self.tail, self.rel_type.strip().upper()))

    def __eq__(self, other):
        if not isinstance(other, RelationRef):
            return False
        return (
            self.head == other.head
            and self.tail == other.tail
            and self.rel_type.strip().upper() == other.rel_type.strip().upper()
        )


@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


# ==================== 工具函数 ====================

def _strip_markdown_json(text: str) -> str:
    """
    去除 LLM 输出中可能包裹的 Markdown 代码块标记（```json ... ```），
    提取纯净的 JSON 字符串。
    """
    text = text.strip()
    # 匹配 ```json ... ``` 或 ``` ... ``` 包裹的代码块
    m = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        logger.debug("检测到 Markdown 代码块包裹，已自动去除")
        return m.group(1).strip()
    return text


def _clean_entities_relations(
    entities: List[Dict], relations: List[Dict]
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    对 LLM 提取的实体和关系做后处理清洗：

    1. 过滤 name 为空的实体
    2. 实体 name 截断：超过20字符的截断并加"..."后缀
    3. 实体 label 白名单校验（大小写不敏感），不在白名单的降级为"其他"
    4. 实体去重：name + label 完全相同的只保留一个
    5. 过滤 head/tail 引用了已被清洗掉的实体的关系

    :return: (cleaned_entities, cleaned_relations, stats_dict)
    """
    stats = {"dropped_empty_name": 0, "name_truncated": 0, "label_fixed": 0, "entity_duplicates": 0, "orphan_relations": 0}
    MAX_NAME_LEN = 20

    # ---- 第一步：逐实体清洗 ----
    cleaned_entities: List[Dict] = []
    seen_entities: Set[Tuple[str, str]] = set()

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        name = str(ent.get("name", "")).strip()
        if not name:
            stats["dropped_empty_name"] += 1
            continue

        # 实体名截断：超过阈值则截取前N字符 + "..."
        if len(name) > MAX_NAME_LEN:
            name = name[:MAX_NAME_LEN] + "..."
            stats["name_truncated"] += 1

        label_raw = str(ent.get("label", "其他")).strip()
        label_upper = label_raw.upper()
        if label_upper not in ALLOWED_ENTITY_LABELS:
            logger.debug(f"实体标签不在白名单: name='{name}', label='{label_raw}'，已修正为'其他'")
            label_upper = "其他"
            stats["label_fixed"] += 1

        # 去重：同 name + 同 label 只保留首次出现
        key = (name, label_upper)
        if key in seen_entities:
            stats["entity_duplicates"] += 1
            continue
        seen_entities.add(key)

        cleaned_ent = {"name": name, "label": label_upper}
        # 保留 description（如果存在）
        desc = str(ent.get("description", "")).strip()
        if desc:
            cleaned_ent["description"] = desc

        cleaned_entities.append(cleaned_ent)

    # ---- 第二步：过滤孤儿关系 ----
    valid_names = {ent["name"] for ent in cleaned_entities}
    cleaned_relations: List[Dict] = []

    for rel in relations:
        if not isinstance(rel, dict):
            continue
        head = str(rel.get("head", "")).strip()
        tail = str(rel.get("tail", "")).strip()
        rel_type = str(rel.get("type", "")).strip()

        if not head or not tail:
            continue
        if head not in valid_names or tail not in valid_names:
            stats["orphan_relations"] += 1
            continue

        cleaned_relations.append({"head": head, "tail": tail, "type": rel_type})

    # 日志
    if any(v > 0 for v in stats.values()):
        logger.debug(
            f"实体关系清洗: 丢弃空名={stats['dropped_empty_name']}, "
            f"截断={stats['name_truncated']}, 标签修正={stats['label_fixed']}, "
            f"去重={stats['entity_duplicates']}, 孤儿关系={stats['orphan_relations']}, "
            f"最终 {len(cleaned_entities)} 实体 / {len(cleaned_relations)} 关系"
        )

    return cleaned_entities, cleaned_relations, stats


# ==================== 预阶段：Milvus 集合准备 + 旧数据清理（仅执行一次） ====================

def _prepare_milvus_collection(state: ImportGraphState, global_item_name: str) -> None:
    """
    准备 Milvus 实体名称集合（不存在则创建），并按 item_name 清理旧数据。
    此函数仅在主线程执行一次，后续各子线程直接写入而无需重复准备。

    :param state: 图状态
    :param global_item_name: 全局产品名称，用于删除旧数据
    """
    entity_collection = milvus_config.entity_name_collection
    if not entity_collection:
        logger.warning("未配置 ENTITY_NAME_COLLECTION，跳过 Milvus 准备")
        return

    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus 客户端不可用，无法写入实体向量。请检查 MILVUS_URL 配置。")

    # 集合不存在则自动创建
    if not milvus_client.has_collection(entity_collection):
        logger.info(f"Milvus 实体名称集合 [{entity_collection}] 不存在，自动创建")
        schema = milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="entity_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            index_name="dense_vector_index",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=entity_collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Milvus 实体名称集合 [{entity_collection}] 创建完成")

    # 按 item_name 删除旧数据（幂等）
    if global_item_name:
        try:
            delete_expr = f"item_name=='{global_item_name}'"
            milvus_client.delete(collection_name=entity_collection, filter=delete_expr)
            logger.info(f"已清理实体名称集合中 item_name='{global_item_name}' 的旧数据")
        except Exception as e:
            logger.warning(f"清理实体名称旧数据异常（可能无旧数据）: {e}")


def _cleanup_neo4j_old_data(global_item_name: str) -> None:
    """
    按 item_name 清理 Neo4j 中的旧图谱数据（节点 + 关系）。
    此函数仅在主线程执行一次，后续 Cypher 批处理不再包含清理语句。

    :param global_item_name: 全局产品名称
    """
    if not global_item_name:
        return

    driver = get_neo4j_driver()
    if driver is None:
        logger.warning("Neo4j 驱动不可用，跳过旧数据清理")
        return

    neo4j_config = get_neo4j_config()
    database = neo4j_config.neo4j_database or "neo4j"

    try:
        with driver.session(database=database) as session:
            session.run(
                "MATCH (n {item_name: $item_name}) DETACH DELETE n",
                item_name=global_item_name,
            )
        logger.info(f"已清理 Neo4j 中 item_name='{global_item_name}' 的旧数据")
    except Exception as e:
        logger.warning(f"清理 Neo4j 旧数据异常（可能无旧数据）: {e}")


# ==================== Step 函数 ====================

@step_log("step_1_validate_get_inputs")
def step_1_validate_get_inputs(state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
    """
    校验并清洗 state 中的 chunks，确保每个 chunk 具备必要字段。

    :param state: 图状态
    :return: (validated_chunks, global_item_name)
    """
    # 1. 获取基础字段
    chunks = state.get("chunks") or []
    global_item_name = str(state.get("item_name", "")).strip()

    # 2. 校验整体 chunks 是否存在
    if not chunks:
        raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

    # 3. 逐个校验 Chunk 的有效性
    validated_chunks = []
    for i, chunk in enumerate(chunks):

        # 3.1 chunk 是否是字典
        if not isinstance(chunk, dict):
            logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
            continue

        # 3.2 处理 chunk_id
        raw_id = chunk.get("chunk_id")
        chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

        # 3.3 获取 content 内容
        content = str(chunk.get("content", "")).strip()
        if not content:
            logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
            continue

        # 3.4 获取 item_name（chunk 级别优先，全局兜底）
        chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
        if not chunk_item:
            logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
            continue

        # 3.5 更新 chunk 字段
        chunk["chunk_id"] = chunk_id
        chunk["item_name"] = chunk_item
        chunk["content"] = content

        # 3.6 加入有效列表
        validated_chunks.append(chunk)

    # 4. 校验清洗后是否还有有效数据
    if not validated_chunks:
        raise ValueError(f"经过清洗后，没有任何有效的 chunk 可用于构建图谱。")

    logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

    return validated_chunks, global_item_name


# ---- 单 Chunk 的 LLM 实体提取（在子线程中调用） ----

def _extract_single_chunk_llm(
    chunk_id: str,
    content: str,
    chunk_item: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    """
    对单个 chunk 调用 LLM（JSON mode）提取实体和关系，含清洗和 3 次重试。

    此函数设计为线程安全——每次调用创建独立的 LLM chain，
    LLM 客户端本身是 HTTP 调用，无本地共享状态。

    :param chunk_id: 切片 ID
    :param content: 切片文本内容
    :param chunk_item: 切片所属产品名称
    :param chunk_index: 切片序号（仅用于日志）
    :param total_chunks: 切片总数（仅用于日志）
    :return: (cleaned_entities, cleaned_relations)
    """
    llm = get_llm_client(json_mode=True)
    parser = JsonOutputParser()
    MAX_RETRIES = 3

    # 构建提示词（不通过 load_prompt 传参，避免 prompt 中 JSON 示例的 {} 被 str.format() 误解析）
    prompt_template = load_prompt("entity_extraction")
    prompt = f"## 输入文本\n{content}\n\n## 所属产品\n{chunk_item}\n\n{prompt_template}"
    messages = [
        SystemMessage(content="你是一个专业的知识图谱构建助手，擅长从设备操作手册中提取实体和关系。"),
        HumanMessage(content=prompt),
    ]

    # 调用 LLM → 提取 content → 清洗 Markdown 包裹 → 解析 JSON
    chain = llm | RunnableLambda(lambda x: x.content) | RunnableLambda(_strip_markdown_json) | parser

    entities, relations = [], []
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = chain.invoke(messages)

            # 校验返回结构
            raw_entities = result.get("entities", []) if isinstance(result, dict) else []
            raw_relations = result.get("relations", []) if isinstance(result, dict) else []

            # 清洗：空名过滤、标签白名单、实体去重、孤儿关系过滤
            entities, relations, clean_stats = _clean_entities_relations(raw_entities, raw_relations)

            if entities or relations:
                break  # 有有效结果，跳出重试
            elif attempt < MAX_RETRIES:
                logger.info(
                    f"Chunk {chunk_id} 第 {attempt} 次提取结果为空，准备重试（共 {MAX_RETRIES} 次机会）"
                )
            else:
                logger.warning(f"Chunk {chunk_id} 经 {MAX_RETRIES} 次尝试仍无有效实体/关系")

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"Chunk {chunk_id} 第 {attempt} 次提取异常: {e}，准备重试（共 {MAX_RETRIES} 次机会）"
                )
            else:
                logger.error(f"Chunk {chunk_id} 经 {MAX_RETRIES} 次尝试均失败: {e}")

    if entities or relations:
        logger.debug(
            f"Chunk {chunk_id} ({chunk_index}/{total_chunks}): "
            f"提取 {len(entities)} 实体 / {len(relations)} 关系"
        )

    return entities, relations


# ---- 单 Chunk 的 Embedding + Milvus 写入（在子线程中调用，内部持有 _embedding_lock） ----

def _embed_and_insert_chunk_entities(
    chunk_id: str,
    entities: List[Dict],
    chunk_item: str,
) -> int:
    """
    为单个 chunk 的实体名称生成稠密+稀疏向量，并写入 Milvus 实体名称集合。

    内部持有 _embedding_lock 互斥锁，确保 BGE-M3 本地模型不会被并发推理。

    :param chunk_id: 切片 ID
    :param entities: 清洗后的实体列表（每项含 name 字段）
    :param chunk_item: 切片所属产品名称
    :return: 写入的记录数
    """
    entity_collection = milvus_config.entity_name_collection
    if not entity_collection:
        return 0

    # 收集本 chunk 的实体名称
    names: List[str] = []
    for ent in entities:
        name = str(ent.get("name", "")).strip()
        if name:
            names.append(name)

    if not names:
        return 0

    # BGE-M3 推理——必须持有互斥锁，防止多线程并发调用导致 PyTorch 内部状态冲突
    with _embedding_lock:
        embeddings = generate_embeddings(names)

    # 组装待插入数据
    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.warning("Milvus 客户端不可用，跳过实体向量写入")
        return 0

    insert_data: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        insert_data.append({
            "entity_name": name,
            "chunk_id": int(chunk_id),
            "item_name": chunk_item,
            "dense_vector": embeddings["dense"][i],
            "sparse_vector": embeddings["sparse"][i],
        })

    result = milvus_client.insert(collection_name=entity_collection, data=insert_data)
    insert_count = result.get("insert_count", 0)
    logger.debug(f"Chunk {chunk_id}: {insert_count} 条实体向量已写入 Milvus")
    return insert_count


# ---- 单 Chunk 的完整处理任务（投入线程池的工作单元） ----

def _process_single_chunk(
    chunk: Dict[str, Any],
    global_item_name: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> Tuple[str, List[Dict], List[Dict], Optional[str]]:
    """
    单个 chunk 的完整处理管线（在子线程中执行）：
    1. LLM 实体关系提取 + 数据清洗
    2. BGE-M3 Embedding + Milvus 写入

    此函数捕获所有异常，不会让线程崩溃。

    :param chunk: 切片字典，需包含 chunk_id / content / item_name
    :param global_item_name: 全局产品名称（兜底）
    :param chunk_index: 切片序号（日志用）
    :param total_chunks: 切片总数（日志用）
    :return: (chunk_id, entities, relations, error_msg)
             error_msg 为 None 表示成功（即使实体/关系为空）
    """
    chunk_id = str(chunk.get("chunk_id", ""))
    chunk_item = str(chunk.get("item_name", global_item_name))
    content = str(chunk.get("content", ""))
    error_msg: Optional[str] = None

    try:
        # Step A: LLM 提取 + 清洗
        entities, relations = _extract_single_chunk_llm(
            chunk_id=chunk_id,
            content=content,
            chunk_item=chunk_item,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

        # Step B: BGE-M3 Embedding + Milvus 写入
        if entities:
            _embed_and_insert_chunk_entities(
                chunk_id=chunk_id,
                entities=entities,
                chunk_item=chunk_item,
            )

        return (chunk_id, entities, relations, error_msg)

    except Exception as e:
        error_msg = f"Chunk {chunk_id} 处理异常: {e}"
        logger.error(error_msg)
        return (chunk_id, [], [], error_msg)


@step_log("step_4_build_cypher")
def step_4_build_cypher(
    state: ImportGraphState,
    extraction_results: List[Tuple[str, List[Dict], List[Dict]]],
    stats: ProcessingStats,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    对提取结果做跨 chunk 去重，构建 Cypher MERGE 语句列表。

    实体以 name + item_name 为标识，同名实体合并标签（Neo4j 多标签）
    和 description（拼接），实现查询收敛。

    注意：Neo4j 旧数据清理已在预阶段由 _cleanup_neo4j_old_data 完成，
    此处不再包含 DETACH DELETE 语句。

    :param state: 图状态
    :param extraction_results: 各 chunk 的提取结果
    :param stats: 处理统计
    :return: cypher_batch: [(cypher_statement, params_dict), ...]
    """
    global_item_name = state.get("item_name", "")

    # ---- 第一遍：按 name 聚合实体的 labels / descriptions / source_chunk_id ----
    # step_2 已保证 name 非空、label 白名单化、chunk 内去重；
    # 此处跨 chunk 聚合同名实体，收集所有 label 和 description
    name_to_labels: Dict[str, Set[str]] = {}
    name_to_descs: Dict[str, List[str]] = {}
    name_to_source_chunk: Dict[str, str] = {}
    all_entity_names: Set[str] = set()

    for chunk_id, entities, relations in extraction_results:
        for ent in entities:
            name = ent["name"]
            label = ent.get("label", "其他")
            desc = ent.get("description", "")

            all_entity_names.add(name)
            if name not in name_to_labels:
                name_to_labels[name] = set()
                name_to_descs[name] = []
                name_to_source_chunk[name] = chunk_id
            name_to_labels[name].add(label)
            if desc and desc not in name_to_descs[name]:
                name_to_descs[name].append(desc)

    # ---- 第二遍：收集 Chunk-Entity 关联 + 关系 ----
    chunk_entity_pairs: Set[Tuple[str, str]] = set()  # (chunk_id, entity_name)
    all_relations: Set[Tuple[str, str, str]] = set()  # (head, tail, rel_type)

    for chunk_id, entities, relations in extraction_results:
        for ent in entities:
            name = ent["name"]
            if name in name_to_labels:
                chunk_entity_pairs.add((chunk_id, name))

        for rel in relations:
            head_name = rel["head"]
            tail_name = rel["tail"]
            rel_type = rel["type"].upper()

            if not rel_type:
                continue
            if rel_type not in ALLOWED_REL_TYPES:
                logger.warning(f"未知关系类型 {rel_type}（head={head_name}, tail={tail_name}），已跳过")
                continue

            # 兜底：关系引用的实体如果不在 entities 列表中，自动补全
            for n in (head_name, tail_name):
                if n not in name_to_labels:
                    name_to_labels[n] = {"其他"}
                    name_to_descs[n] = []
                    name_to_source_chunk[n] = chunk_id
                    all_entity_names.add(n)

            all_relations.add((head_name, tail_name, rel_type))

    stats.total_entities = len(name_to_labels)
    stats.total_relations = len(all_relations)
    logger.info(
        f"聚合完成: {stats.total_entities} 唯一实体（跨 chunk 合并） / {stats.total_relations} 唯一关系, "
        f"{len(chunk_entity_pairs)} 条 Chunk-Entity 关联"
    )

    # ---- 构建 Cypher 语句 ----
    # 注：Neo4j 旧数据清理已由 _cleanup_neo4j_old_data 在预阶段完成

    cypher_batch: List[Tuple[str, Dict[str, Any]]] = []

    # 1. 实体节点
    # label → Neo4j 节点标签（:Entity:DEVICE:PART ...），同名实体合并标签和 description
    for name in sorted(all_entity_names):
        labels = name_to_labels.get(name, {"其他"})
        extra_labels = "".join(f":{l}" for l in sorted(labels))
        descs = name_to_descs.get(name, [])
        desc_merged = "; ".join(descs) if descs else ""
        source_chunk = name_to_source_chunk.get(name, "")

        cypher_batch.append((
            f"""
            MERGE (e:Entity {{name: $name, item_name: $item_name}})
            ON CREATE SET e.source_chunk_id = $source_chunk_id, e.description = $description
            ON MATCH SET e.description = CASE
                WHEN $description = '' THEN e.description
                WHEN e.description IS NULL THEN $description
                WHEN e.description CONTAINS $description THEN e.description
                ELSE e.description + '; ' + $description
            END
            SET e{extra_labels}
            """,
            {
                "name": name, "item_name": global_item_name,
                "source_chunk_id": source_chunk, "description": desc_merged,
            },
        ))

    # 2. Chunk 节点
    seen_chunk_ids: Set[str] = set()
    chunk_item_map: Dict[str, str] = {}
    for ch in state.get("chunks", []):
        cid = str(ch.get("chunk_id", ""))
        if cid:
            chunk_item_map[cid] = str(ch.get("item_name", global_item_name))

    for chunk_id, _, _ in extraction_results:
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        chunk_item = chunk_item_map.get(chunk_id, global_item_name)
        cypher_batch.append((
            """
            MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
            """,
            {"chunk_id": chunk_id, "item_name": chunk_item},
        ))

    # 3. Entity-Chunk 提及关系
    for chunk_id, name in chunk_entity_pairs:
        cypher_batch.append((
            """
            MATCH (e:Entity {name: $name, item_name: $item_name})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            {"chunk_id": chunk_id, "name": name, "item_name": global_item_name},
        ))

    # 4. 实体间关系
    for head_name, tail_name, rel_type in all_relations:
        cypher_batch.append((
            f"""
            MATCH (src:Entity {{name: $src_name, item_name: $item_name}})
            MATCH (tgt:Entity {{name: $tgt_name, item_name: $item_name}})
            MERGE (src)-[:{rel_type}]->(tgt)
            """,
            {
                "src_name": head_name, "tgt_name": tail_name,
                "item_name": global_item_name,
            },
        ))

    logger.info(f"Cypher 语句构建完成，共 {len(cypher_batch)} 条")
    return cypher_batch


@step_log("step_5_execute_neo4j")
def step_5_execute_neo4j(
    cypher_batch: List[Tuple[str, Dict[str, Any]]], stats: ProcessingStats
):
    """
    执行 Cypher 批处理，将图谱数据写入 Neo4j。

    :param cypher_batch: step_4 构建的 Cypher 语句列表
    :param stats: 处理统计（会追加执行错误）
    """
    driver = get_neo4j_driver()
    if driver is None:
        raise RuntimeError("Neo4j 驱动不可用，无法写入图谱数据。请检查 .env 中 NEO4J_URI 等配置。")

    neo4j_config = get_neo4j_config()
    database = neo4j_config.neo4j_database or "neo4j"
    executed = 0
    failed = 0

    with driver.session(database=database) as session:
        for i, (cypher, params) in enumerate(cypher_batch):
            try:
                session.run(cypher, **params)
                executed += 1
            except Exception as e:
                failed += 1
                stats.errors.append(f"Cypher[{i}] 执行失败: {str(e)}")
                logger.warning(f"Cypher[{i}] 执行失败: {e}")

    logger.info(f"Neo4j 写入完成: {executed} 成功 / {failed} 失败")
    if failed > 0:
        logger.warning(f"有 {failed} 条 Cypher 执行失败，详情见 stats.errors")


@step_log("step_6_update_state")
def step_6_update_state(state: ImportGraphState, stats: ProcessingStats):
    """
    将图谱构建结果写回 state。

    :param state: 图状态
    :param stats: 处理统计
    """
    state["kg_stats"] = {
        "total_chunks": stats.total_chunks,
        "processed_chunks": stats.processed_chunks,
        "failed_chunks": stats.failed_chunks,
        "total_entities": stats.total_entities,
        "total_relations": stats.total_relations,
        "errors": stats.errors,
    }
    state["kg_id"] = state.get("task_id", "")
    logger.info(f"图谱状态已更新: kg_id={state['kg_id']}, {stats.summary()}")


# ==================== LangGraph 入口节点 ====================

# 线程池最大并发数——LLM 提取以 API 调用为主（IO 密集型），并发数可适当调高；
# 但 BGE-M3 Embedding 受互斥锁序列化，实际并发收益主要来自 LLM 阶段的重叠调用
_MAX_WORKERS = 4


@node_log("node_import_neo4j")
def node_import_neo4j(state: ImportGraphState) -> ImportGraphState:
    """
    节点功能：从文档切块中提取实体和关系，构建知识图谱并写入 Neo4j，
    同时将实体名称向量化存入 Milvus 实体名称集合，支持后续实体级混合检索。

    多线程执行流程：
      ┌─ 预阶段（主线程，仅一次）────────────────────────────┐
      │  · 校验 chunks · 准备 Milvus 集合 · 清理 Milvus/Neo4j 旧数据 │
      └──────────────────────────────────────────────────────┘
                              ↓
      ┌─ 并发阶段（线程池，每个 chunk 一个任务）──────────────┐
      │  每个子线程: LLM 提取+清洗 → BGE-M3 Embedding(互斥锁) │
      │              → Milvus 写入 → 返回 (实体, 关系)        │
      └──────────────────────────────────────────────────────┘
                              ↓
      ┌─ 收尾阶段（主线程，等待所有线程完成）─────────────────┐
      │  跨 chunk 聚合去重 → 构建 Cypher → Neo4j 写入 → 更新 state │
      └──────────────────────────────────────────────────────┘

    前置依赖：
      - state["chunks"] 中每个 chunk 须包含 chunk_id（由 node_import_milvus 回写）
      - state["item_name"] / state["file_title"] / state["task_id"]

    产出：
      - Neo4j 中的 Chunk 节点 + 多标签 Entity 节点（:Entity:DEVICE:...）
        及 MENTIONED_IN / 8种实体间关系
      - Milvus 实体名称集合中的实体稠密+稀疏向量（可按实体名做混合检索）
      - state["kg_id"] / state["kg_stats"]
    """
    # 任务开始
    add_running_task(state["task_id"], "node_import_neo4j")

    # ---- 第一阶段：参数校验 ----
    validated_chunks, global_item_name = step_1_validate_get_inputs(state)
    total = len(validated_chunks)

    # ---- 第二阶段：预准备（主线程，仅一次） ----
    # 准备 Milvus 集合 + 清理 Milvus 旧数据
    _prepare_milvus_collection(state, global_item_name)

    # 清理 Neo4j 旧数据
    _cleanup_neo4j_old_data(global_item_name)

    # ---- 第三阶段：多线程并发处理每个 chunk ----
    stats = ProcessingStats(total_chunks=total)
    extraction_results: List[Tuple[str, List[Dict], List[Dict]]] = []

    # 根据 chunk 数量动态调整线程池大小
    max_workers = min(_MAX_WORKERS, total) if total > 0 else 1
    logger.info(f"启动线程池（max_workers={max_workers}），开始并发处理 {total} 个切片")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_chunk = {
            executor.submit(
                _process_single_chunk,
                chunk=chunk,
                global_item_name=global_item_name,
                chunk_index=i + 1,
                total_chunks=total,
            ): chunk
            for i, chunk in enumerate(validated_chunks)
        }

        # 按完成顺序收集结果
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            chunk_id = str(chunk.get("chunk_id", "unknown"))
            try:
                cid, entities, relations, error_msg = future.result()
                if error_msg:
                    stats.failed_chunks += 1
                    stats.errors.append(error_msg)
                elif entities or relations:
                    extraction_results.append((cid, entities, relations))
                    stats.processed_chunks += 1
                else:
                    stats.failed_chunks += 1
                    stats.errors.append(f"Chunk {cid} 提取结果为空")
                    logger.warning(f"Chunk {cid} 未提取到任何实体/关系")
            except Exception as e:
                stats.failed_chunks += 1
                stats.errors.append(f"Chunk {chunk_id} 线程异常: {e}")
                logger.error(f"Chunk {chunk_id} 线程异常: {e}")

    # Milvus 批量 flush，确保所有子线程写入的数据持久化
    entity_collection = milvus_config.entity_name_collection
    if entity_collection:
        try:
            milvus_client = get_milvus_client()
            if milvus_client:
                milvus_client.flush(collection_name=entity_collection)
                logger.debug(f"Milvus 集合 [{entity_collection}] flush 完成")
        except Exception as e:
            logger.warning(f"Milvus flush 异常: {e}")

    logger.info(f"多线程处理完成: {stats.processed_chunks} 成功 / {stats.failed_chunks} 失败")

    # ---- 第四阶段：构建 Cypher 并写入 Neo4j ----
    if not extraction_results:
        logger.warning("所有 chunk 均未提取到实体/关系，跳过 Neo4j 写入")
        stats.errors.append("所有 chunk 均未提取到有效实体/关系")
    else:
        cypher_batch = step_4_build_cypher(state, extraction_results, stats)
        step_5_execute_neo4j(cypher_batch, stats)

    # ---- 第五阶段：更新 state ----
    step_6_update_state(state, stats)

    # 任务结束
    add_done_task(state["task_id"], "node_import_neo4j")
    return state


# ==================== 本地测试入口 ====================

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("===== 开始执行 node_import_neo4j 本地测试 =====")

    # 构造模拟 state（模拟 Milvus 入库后的 chunks，已带 chunk_id）
    test_state: ImportGraphState = {
        "task_id": "test_kg_neo4j_001",
        "file_title": "HAK180产品安全手册",
        "item_name": "HAK 180 烫金机",
        "chunks": [
            {
                "chunk_id": 1001,
                "content": "烫金机HAK 180适用于多种材料的烫金加工，包括纸张、皮革、塑料等。设备采用微电脑控制系统，具备自动温控功能。配备漏电保护装置，检测到漏电电流超过一定数值时自动断电。",
                "title": "# 产品概述",
                "file_title": "HAK180产品安全手册",
                "item_name": "HAK 180 烫金机",
                "part": 1,
            },
            {
                "chunk_id": 1002,
                "content": "电源规格：额定电压220V/50Hz，功率2000W。配备漏电保护装置，当检测到漏电电流超过30mA时自动断电。",
                "title": "# 技术参数",
                "file_title": "HAK180产品安全手册",
                "item_name": "HAK 180 烫金机",
                "part": 2,
            },
            {
                "chunk_id": 1003,
                "content": "日常维护：定期清理烫金板表面残留物，检查温度传感器是否正常。每工作500小时需更换导热硅脂，确保加热均匀。",
                "title": "# 维护保养",
                "file_title": "HAK180产品安全手册",
                "item_name": "HAK 180 烫金机",
                "part": 3,
            },
        ],
    }

    try:
        # 校验 Neo4j 可用性
        driver = get_neo4j_driver()
        if driver is None:
            logger.error("Neo4j 不可用，测试终止。请检查 .env 配置或启动 Neo4j 服务。")
        else:
            # 注意：不要在这里 driver.close()，因为 step_5 会通过单例获取同一个 driver
            result_state = node_import_neo4j(test_state)
            kg_stats = result_state.get("kg_stats", {})
            logger.info(f"===== 测试完成 =====")
            logger.info(f"kg_id: {result_state.get('kg_id')}")
            logger.info(f"实体数: {kg_stats.get('total_entities')}, 关系数: {kg_stats.get('total_relations')}")
            logger.info(f"成功/失败切片: {kg_stats.get('processed_chunks')}/{kg_stats.get('failed_chunks')}")
            if kg_stats.get("errors"):
                logger.warning(f"错误列表: {kg_stats['errors']}")
    except Exception as e:
        logger.exception(f"===== node_import_neo4j 测试失败: {e} =====")
