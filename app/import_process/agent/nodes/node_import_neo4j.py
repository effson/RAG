# 导入基础库：系统、JSON解析、类型注解
import sys
import json
import re
import os
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field

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


# ---- 数据结构定义 ----

@dataclass
class EntityRef:
    """去重用的实体引用，以(name, label)为唯一标识"""
    name: str
    label: str

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


@step_log("step_2_extract_entities_relations")
def step_2_extract_entities_relations(
    validated_chunks: List[Dict[str, Any]], global_item_name: str
) -> Tuple[List[Tuple[str, List[Dict], List[Dict]]], ProcessingStats]:
    """
    对每个 chunk 调用 LLM（JSON mode），提取实体和关系。

    :param validated_chunks: 校验后的 chunk 列表
    :param global_item_name: 全局产品名称
    :return:
    """
    llm = get_llm_client(json_mode=True)
    parser = JsonOutputParser()
    extraction_results: List[Tuple[str, List[Dict], List[Dict]]] = []
    stats = ProcessingStats(total_chunks=len(validated_chunks))
    MAX_RETRIES = 3

    for i, chunk in enumerate(validated_chunks):
        chunk_id = chunk["chunk_id"]
        chunk_item = chunk.get("item_name", global_item_name)
        content = chunk["content"]

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
            extraction_results.append((chunk_id, entities, relations))
            stats.processed_chunks += 1
            logger.debug(
                f"Chunk {chunk_id} ({i+1}/{len(validated_chunks)}): "
                f"提取 {len(entities)} 实体 / {len(relations)} 关系"
            )
        else:
            stats.failed_chunks += 1
            if last_error:
                stats.errors.append(f"Chunk {chunk_id} 提取失败: {str(last_error)}")
            else:
                stats.errors.append(f"Chunk {chunk_id} 提取失败: {MAX_RETRIES} 次尝试后仍无有效实体/关系")

    logger.info(f"实体提取完成: {stats.processed_chunks} 成功 / {stats.failed_chunks} 失败")
    return extraction_results, stats


@step_log("step_4_build_cypher")
def step_4_build_cypher(
    state: ImportGraphState,
    extraction_results: List[Tuple[str, List[Dict], List[Dict]]],
    stats: ProcessingStats,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    对提取结果做跨 chunk 去重，构建 Cypher MERGE 语句列表。

    :param state: 图状态（获取 file_title / task_id / item_name）
    :param extraction_results: step_2 的提取结果
    :param stats: 处理统计（会更新 total_entities / total_relations）
    :return: cypher_batch: [(cypher_statement, params_dict), ...]
    """
    file_title = state.get("file_title", "")
    task_id = state.get("task_id", "")
    global_item_name = state.get("item_name", "")

    # ---- 第一遍：跨 chunk 去重实体，构建 name→EntityRef 映射 ----
    # 注：step_2 已保证实体 name 非空、label 白名单化、chunk 内去重；
    #     此处只需做跨 chunk 去重（同名实体首次 label 优先）
    all_entities: Set[EntityRef] = set()
    name_to_entity: Dict[str, EntityRef] = {}  # name → EntityRef（首遇优先）

    for chunk_id, entities, relations in extraction_results:
        for ent in entities:
            name = ent["name"]  # step_2 已保证非空
            label = ent.get("label", "其他")
            eref = EntityRef(name=name, label=label)
            all_entities.add(eref)
            if name not in name_to_entity:
                name_to_entity[name] = eref

    # ---- 第二遍：收集 Chunk-Entity 关联 + 关系 ----
    # 注：step_2 已保证关系 head/tail 非空且非孤儿；此处只需校验关系类型白名单
    chunk_entity_pairs: Set[Tuple[str, EntityRef]] = set()
    all_relations: Set[RelationRef] = set()

    for chunk_id, entities, relations in extraction_results:
        # Chunk-Entity 关联
        for ent in entities:
            name = ent["name"]
            if name in name_to_entity:
                chunk_entity_pairs.add((chunk_id, name_to_entity[name]))

        # 关系：校验类型白名单（step_2 未做此检查）+ 补全关系引用的实体
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
            head_ref = name_to_entity.get(head_name)
            if head_ref is None:
                head_ref = EntityRef(name=head_name, label="其他")
                all_entities.add(head_ref)
                name_to_entity[head_name] = head_ref

            tail_ref = name_to_entity.get(tail_name)
            if tail_ref is None:
                tail_ref = EntityRef(name=tail_name, label="其他")
                all_entities.add(tail_ref)
                name_to_entity[tail_name] = tail_ref

            all_relations.add(RelationRef(head=head_ref, tail=tail_ref, rel_type=rel_type))

    stats.total_entities = len(all_entities)
    stats.total_relations = len(all_relations)
    logger.info(
        f"去重完成: {stats.total_entities} 唯一实体 / {stats.total_relations} 唯一关系, "
        f"{len(chunk_entity_pairs)} 条 Chunk-Entity 关联"
    )

    # ---- 构建 Cypher 语句 ----

    cypher_batch: List[Tuple[str, Dict[str, Any]]] = []

    # 1. 实体节点（仅以 name 为唯一标识）
    for eref in all_entities:
        cypher_batch.append((
            """
            MERGE (e:Entity {name: $name})
            """,
            {"name": eref.name},
        ))

    # 2. Chunk 节点（id + item_name，不再挂载到 Document）
    seen_chunk_ids: Set[str] = set()
    # chunk_id → item_name 快速查找表
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
            MERGE (c:Chunk {id: $chunk_id})
            SET c.item_name = $item_name
            """,
            {"chunk_id": chunk_id, "item_name": chunk_item},
        ))

    # 3. Entity-Chunk 提及关系（实体被提及于哪个 Chunk）
    for chunk_id, eref in chunk_entity_pairs:
        cypher_batch.append((
            """
            MATCH (e:Entity {name: $name})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            {"chunk_id": chunk_id, "name": eref.name},
        ))

    # 4. 实体间关系（使用 LLM 指定的关系类型作为 Neo4j 关系 Type）
    for rref in all_relations:
        rel_type = rref.rel_type  # 已通过白名单校验，可直接用于 Cypher
        cypher_batch.append((
            f"""
            MATCH (src:Entity {{name: $src_name}})
            MATCH (tgt:Entity {{name: $tgt_name}})
            MERGE (src)-[:{rel_type}]->(tgt)
            """,
            {
                "src_name": rref.head.name,
                "tgt_name": rref.tail.name,
            },
        ))

    logger.info(f"Cypher 语句构建完成，共 {len(cypher_batch)} 条")
    return cypher_batch


@step_log("step_3_import_entity_vectors")
def step_3_import_entity_vectors(
    state: ImportGraphState,
    extraction_results: List[Tuple[str, List[Dict], List[Dict]]],
) -> None:
    """
    将提取的实体名称通过 BGE-M3 生成稠密+稀疏向量，存入 Milvus 实体名称集合。

    每个 (实体名, chunk_id) 对存储一条记录，支持按 chunk_id 追溯实体来源，
    后续查询时可按实体名做稠密+稀疏混合检索，定位相关实体所在的文档切片。

    :param state: 图状态（获取 item_name 等全局字段）
    :param extraction_results: step_2 的提取结果 [(chunk_id, entities, relations), ...]
    """
    entity_collection = milvus_config.entity_name_collection
    if not entity_collection:
        logger.warning("未配置 ENTITY_NAME_COLLECTION，跳过实体向量入库")
        return

    # ---- 第一步：收集所有 (实体名, chunk_id, item_name) 三元组 ----
    entity_rows: List[Dict[str, Any]] = []  # {entity_name, chunk_id, item_name}
    seen_pairs: Set[Tuple[str, str]] = set()  # (entity_name, chunk_id) 去重

    # 构建 chunk_id → item_name 快速查找表
    chunk_item_map: Dict[str, str] = {}
    for ch in state.get("chunks", []):
        cid = str(ch.get("chunk_id", ""))
        if cid:
            chunk_item_map[cid] = str(ch.get("item_name", ""))

    global_item_name = str(state.get("item_name", "")).strip()

    for chunk_id, entities, _ in extraction_results:
        chunk_id_str = str(chunk_id)
        chunk_item = chunk_item_map.get(chunk_id_str, global_item_name)
        if not chunk_item:
            chunk_item = global_item_name

        for ent in entities:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            key = (name, chunk_id_str)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            entity_rows.append({
                "entity_name": name,
                "chunk_id": int(chunk_id),
                "item_name": chunk_item,
            })

    if not entity_rows:
        logger.info("没有实体需要存入 Milvus 实体名称集合")
        return

    logger.info(f"准备为 {len(entity_rows)} 条 (实体, chunk) 对生成向量并入库")

    # ---- 第二步：准备 Milvus 集合（获取客户端 + 不存在则创建） ----
    milvus_client = get_milvus_client()
    if milvus_client is None:
        raise RuntimeError("Milvus 客户端不可用，无法写入实体向量。请检查 MILVUS_URL 配置。")

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

    # ---- 第三步：按 item_name 删除旧数据（幂等，在生成向量前清理） ----
    if global_item_name:
        try:
            delete_expr = f"item_name=='{global_item_name}'"
            milvus_client.delete(collection_name=entity_collection, filter=delete_expr)
            logger.info(f"已清理实体名称集合中 item_name='{global_item_name}' 的旧数据")
        except Exception as e:
            logger.warning(f"清理实体名称旧数据异常（可能无旧数据）: {e}")

    # ---- 第四步：调用 BGE-M3 生成稠密+稀疏向量 ----
    entity_names = [row["entity_name"] for row in entity_rows]
    embeddings = generate_embeddings(entity_names)

    # ---- 第五步：组装待插入数据（向量附加到每一行） ----
    insert_data: List[Dict[str, Any]] = []
    for i, row in enumerate(entity_rows):
        insert_data.append({
            "entity_name": row["entity_name"],
            "chunk_id": row["chunk_id"],
            "item_name": row["item_name"],
            "dense_vector": embeddings["dense"][i],
            "sparse_vector": embeddings["sparse"][i],
        })

    # ---- 第六步：批量插入 ----
    result = milvus_client.insert(collection_name=entity_collection, data=insert_data)
    insert_count = result.get("insert_count", 0)
    milvus_client.flush(collection_name=entity_collection)
    logger.info(f"实体向量入库完成: {insert_count} 条记录 → Milvus集合[{entity_collection}]")


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

@node_log("node_import_neo4j")
def node_import_neo4j(state: ImportGraphState) -> ImportGraphState:
    """
    节点功能：从文档切块中提取实体和关系，构建知识图谱并写入 Neo4j，
    同时将实体名称向量化存入 Milvus 实体名称集合，支持后续实体级混合检索。

    前置依赖：
      - state["chunks"] 中每个 chunk 须包含 chunk_id（由 node_import_milvus 回写）
      - state["item_name"] / state["file_title"] / state["task_id"]

    产出：
      - Neo4j 中的 Document / ItemName / Chunk / Entity 节点及关系
      - Milvus 实体名称集合中的实体稠密+稀疏向量（可按实体名做混合检索）
      - state["kg_id"] / state["kg_stats"]
    """
    # 日志+任务处理
    add_running_task(state["task_id"], "node_import_neo4j")

    # 1. 参数校验与清洗
    validated_chunks, global_item_name = step_1_validate_get_inputs(state)

    # 2. LLM 实体关系提取
    extraction_results, stats = step_2_extract_entities_relations(
        validated_chunks, global_item_name
    )

    # 3. 实体名称向量化并存入 Milvus
    step_3_import_entity_vectors(state, extraction_results)

    # 4. 去重 + 构建 Cypher 语句
    cypher_batch = step_4_build_cypher(state, extraction_results, stats)

    # 5. 执行 Cypher 写入 Neo4j
    step_5_execute_neo4j(cypher_batch, stats)

    # 6. 更新 state
    step_6_update_state(state, stats)

    # 日志+任务处理
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
                "content": "烫金机HAK 180适用于多种材料的烫金加工，包括纸张、皮革、塑料等。设备采用微电脑控制系统，具备自动温控功能。",
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
