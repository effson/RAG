# KG（知识图谱）检索节点：利用 Neo4j 中的实体关系图，通过实体名匹配和图遍历定位相关切片
import collections
import re
import sys
from typing import List, Dict, Any, Optional, Tuple, Set

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.neo4j_utils import get_neo4j_driver
from app.conf.neo4j_conf import get_config as get_neo4j_config
from app.clients.milvus_utils import (
    get_milvus_client, fetch_chunks_by_chunk_ids,
    create_hybrid_search_requests, hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# ---- 配置常量 ----
# KG 路在 RRF 融合时的权重
KG_WEIGHT = 1.0

# 实体名最大长度（与导入侧 entity_extraction.prompt 保持一致）
MAX_ENTITY_NAME_LENGTH = 15

# Milvus 实体名对齐的最低混合检索分数阈值（norm_score 归一化后范围 0~1）
# 低于此阈值视为未命中，宁缺勿滥
ENTITY_ALIGN_SCORE_THRESHOLD = 0.50

# 图谱实体类型中文描述（与 entity_recognition.prompt 中的 {allow_entity_labels_cn} 对应）
ALLOW_ENTITY_LABELS_CN = "Device(设备), Part(部件), Operation(操作), Step(步骤), Warning(警告), Condition(条件), Tool(工具)"

# ---- Neo4j 图搜索配置 ----
# 种子节点数量上限（防止后续一跳查询笛卡尔积爆炸）
SEED_NODE_MAX = 20

# CONTAINS 模糊匹配每实体最多取几条
CONTAINS_MATCH_LIMIT = 3

# MENTIONED_IN Chunk 打分权重
SEED_CHUNK_WEIGHT = 2
NEIGHBOR_CHUNK_WEIGHT = 1


@step_log("step_1_data_validates")
def step_1_data_validates(state: QueryGraphState):
    """
    参数校验 + 从 rewritten_query 中剔除已确认的商品名。

    设计考量：
    - 导入侧切片文档中通常不包含设备主语，图谱里也没有把商品名作为 Entity 节点
    - item_name 过滤已在后续 Milvus expr 和 Neo4j Cypher WHERE 子句中生效
    - 提前剥离商品名可避免后续 LLM 提取出无用的品牌/型号关键词

    :param state: 图状态
    :return: (item_names, cleaned_query)
    """
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if not item_names or not rewritten_query:
        logger.error("item_names或rewritten_query不存在,无法继续KG检索!")
        raise ValueError("item_names或rewritten_query不存在,无法继续KG检索!")

    # 从问题文本中移除商品名：按长度降序替换，避免短名截断长名
    cleaned = rewritten_query
    for name in sorted(item_names, key=len, reverse=True):
        cleaned = cleaned.replace(name, "")
    # 清理替换后产生的多余空白
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    logger.info(f"KG检索 - 剥离商品名: '{rewritten_query}' → '{cleaned}'")
    return item_names, cleaned


@step_log("step_2_extract_keywords")
def step_2_extract_keywords(cleaned_query: str) -> List[str]:
    """
    调用 LLM 从剥离商品名的问题中提取实体名，用于 Neo4j 实体 CONTAINS 匹配。

    清洗流程（与导入侧 entity_extraction 对齐）：
        1. 加载外部 prompt 模板 (entity_recognition.prompt)
        2. LLM (JSON mode) 提取实体名
        3. _strip_markdown_json: 去除可能的 Markdown 围栏
        4. JSON 反序列化 → 取 entities 数组
        5. 实体名截断至 {MAX_ENTITY_NAME_LENGTH} 字符
        6. 去重

    降级策略：LLM 调用或 JSON 解析失败 → 返回空列表，不阻断主流程

    :param cleaned_query: 已剥离商品名的问题文本
    :return: 清洗后的实体名列表
    """
    llm = get_llm_client(json_mode=True)
    parser = JsonOutputParser()

    # 从外部 prompt 模板加载提示词
    prompt = load_prompt(
        "entity_recognition",
        allow_entity_labels_cn=ALLOW_ENTITY_LABELS_CN,
        MAX_ENTITY_NAME_LENGTH=str(MAX_ENTITY_NAME_LENGTH),
        query=cleaned_query,
    )

    messages = [
        SystemMessage(content="你是知识图谱系统的实体识别领域专家，擅长从用户问题中提取可用于图谱查询的实体名称。"),
        HumanMessage(content=prompt),
    ]

    try:
        # LLM → 提取 content → 清洗 Markdown 围栏 → JSON 解析
        chain = llm | RunnableLambda(lambda x: x.content) | RunnableLambda(_strip_markdown_json) | parser
        result = chain.invoke(messages)
        raw_entities = result.get("entities", []) if isinstance(result, dict) else []
    except Exception as e:
        logger.warning(f"KG检索 - 实体名提取失败: {e}，返回空列表")
        return []

    # 清洗：截断 + 去重
    seen: set = set()
    cleaned: List[str] = []
    for name in raw_entities:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        if len(name) > MAX_ENTITY_NAME_LENGTH:
            name = name[:MAX_ENTITY_NAME_LENGTH] + "..."
        if name not in seen:
            seen.add(name)
            cleaned.append(name)

    logger.info(f"KG检索 - 提取到实体名: {cleaned}")
    return cleaned


def _strip_markdown_json(text: str) -> str:
    """
    去除 LLM 输出中可能包裹的 Markdown 代码块标记（```json ... ```），
    提取纯净的 JSON 字符串。与 node_import_neo4j 中同名函数逻辑一致。
    """
    text = text.strip()
    m = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        logger.debug("KG检索 - 检测到 Markdown 代码块包裹，已自动去除")
        return m.group(1).strip()
    return text


def _dedup_entity_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """去重：(entity_name, item_name) 都相同的视为重复。"""
    seen: set = set()
    result: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


@step_log("step_3_align_entities_with_milvus")
def step_3_align_entities_with_milvus(
    llm_entities: List[str], item_names: List[str]
) -> List[Tuple[str, str]]:
    """
    将 LLM 提取的原始实体名与 Milvus ENTITY_NAME_COLLECTION 中已入库的
    标准实体名做 BGE-M3 混合检索语义对齐。

    算法（逐 item_name 检索）：
        对每个 item_name，以该 item_name 为过滤条件，对所有 llm_entity 逐一
        做 BGE-M3 混合检索取 top1：
          - 分数 >= ENTITY_ALIGN_SCORE_THRESHOLD → 采用 Milvus 标准实体名
          - 分数不足 / item_name 不匹配 / 无结果 / 异常 → 回退使用 LLM 原始实体名
        最终每个对齐结果都是一个 (entity_name, item_name) 对。

    设计考量：
    - LLM 提取的实体名是自然语言描述（如"漏电保护"），图谱中存储的是
      导入侧从文档原文抽取的标准实体名（如"漏电保护装置"），两者措辞可能不同
    - 通过 BGE-M3 稠密+稀疏双向量混合检索，在语义空间中找到最接近的标准实体名
    - 按 item_name 分别检索，确保每个实体的 item_name 归属明确
    - 分数阈值过滤低质量对齐，未达标者以 LLM 原始名兜底（宁滥勿缺，给 Neo4j CONTAINS 匹配留机会）

    降级策略：Milvus 不可用 / embedding 失败 → 返回 LLM 实体 × item_names 笛卡尔积

    :param llm_entities: LLM 提取的原始实体名列表
    :param item_names: 已确认的商品名列表
    :return: 已去重的 (entity_name, item_name) 对列表
    """
    if not llm_entities:
        return []

    entity_collection = milvus_config.entity_name_collection
    milvus_client = get_milvus_client()

    # 降级：Milvus 不可用 → LLM 实体 × item_names 笛卡尔积
    if not entity_collection or milvus_client is None:
        logger.warning("Milvus 不可用，使用 LLM 原始实体 × item_names 笛卡尔积")
        pairs: List[Tuple[str, str]] = []
        for iname in item_names:
            for ent in llm_entities:
                pairs.append((ent, iname))
        return _dedup_entity_pairs(pairs)

    # 3a. BGE-M3 批量向量化：一次性将所有 LLM 实体名转为 dense + sparse 向量
    #     与导入侧 node_import_neo4j 使用同一个 BGE-M3 模型，向量空间一致
    try:
        embeddings = generate_embeddings(llm_entities)
    except Exception as e:
        logger.warning(f"实体名向量化失败: {e}，使用 LLM 原始实体 × item_names 笛卡尔积")
        pairs: List[Tuple[str, str]] = []
        for iname in item_names:
            for ent in llm_entities:
                pairs.append((ent, iname))
        return _dedup_entity_pairs(pairs)

    # 3b. 逐 item_name 检索：每个 item_name 独立过滤，确保实体归属明确
    aligned: List[Tuple[str, str]] = []  # (entity_name, item_name)
    skipped = 0

    for item_name in item_names:
        expr_str = f'item_name == "{item_name}"'

        for i, llm_name in enumerate(llm_entities):
            dense_vector = embeddings["dense"][i]
            sparse_vector = embeddings["sparse"][i]

            try:
                reqs = create_hybrid_search_requests(
                    dense_vector, sparse_vector, expr=expr_str, limit=1,
                )
                resp = hybrid_search(
                    client=milvus_client,
                    collection_name=entity_collection,
                    reqs=reqs,
                    ranker_weights=(0.4, 0.6),
                    norm_score=True,
                    limit=1,
                    output_fields=["entity_name", "item_name"],
                )
            except Exception as e:
                logger.warning(
                    f"实体 '{llm_name}' (item={item_name}) 检索异常: {e}，回退使用 LLM 原始名"
                )
                aligned.append((llm_name, item_name))
                continue

            # 解析 top1 结果并校验：分数 + 实体名 + item_name 三重门
            if resp and len(resp) > 0 and len(resp[0]) > 0:
                top1 = resp[0][0]
                score = top1.get("distance", 0)
                entity = top1.get("entity", {})
                matched_name = entity.get("entity_name", "")
                matched_item_name = entity.get("item_name", "")

                if (
                    score >= ENTITY_ALIGN_SCORE_THRESHOLD
                    and matched_name
                    and matched_item_name == item_name
                ):
                    logger.debug(
                        f"实体对齐 ✓: '{llm_name}' → '{matched_name}' "
                        f"(item={item_name}, score={score:.4f})"
                    )
                    aligned.append((matched_name, item_name))
                else:
                    logger.debug(
                        f"实体对齐 ✗: '{llm_name}' (item={item_name}) "
                        f"score={score:.4f}，回退使用 LLM 原始名"
                    )
                    aligned.append((llm_name, item_name))
                    skipped += 1
            else:
                logger.debug(
                    f"实体对齐 ✗: '{llm_name}' (item={item_name}) 无匹配结果，回退使用 LLM 原始名"
                )
                aligned.append((llm_name, item_name))
                skipped += 1

    # 3c. 去重：(entity_name, item_name) 都相同才视为重复
    result = _dedup_entity_pairs(aligned)

    logger.info(
        f"KG检索 - 实体对齐: {len(llm_entities)} LLM实体 × {len(item_names)} 商品 "
        f"→ {len(result)} 个 (entity, item_name) 对（跳过 {skipped} 个低分/无匹配）"
    )
    return result


@step_log("step_4_neo4j_graph_search")
def step_4_neo4j_graph_search(
    entity_item_pairs: List[Tuple[str, str]],
    item_names: List[str],
) -> tuple:
    """
    在 Neo4j 中执行图搜索，五个子步骤：

    4a. 种子节点定位：对每个 (entity_name, item_name) 对，先精确匹配 Entity 节点，
        找不到时 CONTAINS 模糊降级；种子总数有 SEED_NODE_MAX 上限。
    4b. 一跳关系扩展：从种子节点双向查询邻居（过滤 MENTIONED_IN），
        保留真实方向（head→tail 与图谱一致），跨种子边去重。
    4c. MENTIONED_IN 反查 Chunk 加权打分：种子提及的 Chunk 得 2 分，
        一跳邻居提及的 Chunk 得 1 分；按 (得分, 提及次数) 降序排列。
    4d. Milvus 批量回填：按排序后的 chunk_id 列表从 Milvus 批量获取切片文本，
        结果按加权得分降序重排。
    4e. 三元组转文本：将一跳实体关系转为 LLM 易理解的文本描述。

    回退策略：entity_item_pairs 为空时，以 item_name 下全部 Entity 为种子。

    :param entity_item_pairs: (实体名, 商品名) 对列表，来自 step_3 对齐结果
    :param item_names: 商品名列表（entity_item_pairs 为空时的回退用）
    :return: (kg_chunks: List[Dict], entity_relations: List[Dict], kg_triple_text: str)
             kg_chunks 为 RRF 兼容格式 [{"id": ..., "entity": {...}}, ...]，已按得分降序
             entity_relations 为实体关系 [{source, relation, target}]
             kg_triple_text 为三元组文本描述（供 answer_output prompt 拼装）
    """
    driver = get_neo4j_driver()
    if driver is None:
        logger.warning("Neo4j 驱动不可用，跳过 KG 图搜索")
        return [], [], ""

    neo4j_config = get_neo4j_config()
    database = neo4j_config.neo4j_database or "neo4j"

    # ---- 4a. 种子节点定位 ----
    seed_entities: Set[Tuple[str, str]] = set()  # (name, item_name)

    with driver.session(database=database) as session:
        if entity_item_pairs:
            for entity_name, item_name in entity_item_pairs:
                if len(seed_entities) >= SEED_NODE_MAX:
                    break

                # 4a-i. 精确匹配：(name, item_name) 组合在导入侧 MERGE 写入，事实上唯一
                try:
                    exact_result = session.run(
                        """
                        MATCH (e:Entity {name: $name, item_name: $item_name})
                        RETURN e.name AS name, e.item_name AS item_name
                        LIMIT 1
                        """,
                        name=entity_name, item_name=item_name,
                    )
                    exact_records = list(exact_result)
                    if exact_records:
                        for rec in exact_records:
                            seed_entities.add((rec["name"], rec["item_name"]))
                        logger.debug(f"种子节点(精确): '{entity_name}' @ {item_name}")
                        continue
                except Exception as e:
                    logger.warning(f"种子精确匹配异常 ('{entity_name}' @ {item_name}): {e}")

                # 4a-ii. CONTAINS 模糊降级
                try:
                    fuzzy_result = session.run(
                        """
                        MATCH (e:Entity {item_name: $item_name})
                        WHERE e.name CONTAINS $name
                        RETURN e.name AS name, e.item_name AS item_name
                        LIMIT $limit
                        """,
                        name=entity_name, item_name=item_name,
                        limit=CONTAINS_MATCH_LIMIT,
                    )
                    fuzzy_count = 0
                    for rec in fuzzy_result:
                        if len(seed_entities) >= SEED_NODE_MAX:
                            break
                        seed_entities.add((rec["name"], rec["item_name"]))
                        fuzzy_count += 1
                    if fuzzy_count > 0:
                        logger.debug(
                            f"种子节点(模糊): '{entity_name}' → {fuzzy_count} 个 @ {item_name}"
                        )
                    else:
                        logger.debug(f"种子节点 ✗: '{entity_name}' @ {item_name} 无匹配")
                except Exception as e:
                    logger.warning(f"种子模糊匹配异常 ('{entity_name}' @ {item_name}): {e}")

        else:
            # entity_item_pairs 为空表示 LLM 未提取到任何实体，无需继续
            logger.info("KG 图搜索: entity_item_pairs 为空，跳过")
            return [], [], ""

    if not seed_entities:
        logger.info("KG 图搜索: 未找到任何种子节点")
        return [], [], ""

    logger.info(f"KG 图搜索 - 4a 种子节点: {len(seed_entities)} 个")

    # ---- 4b. 一跳关系扩展 ----
    entity_relations: List[Dict] = []
    neighbor_entities: Set[Tuple[str, str]] = set()  # (name, item_name)
    seen_edges: Set[Tuple[str, str, str, str, str]] = set()
    # edge key: (head_name, head_item, rel_type, tail_name, tail_item)

    with driver.session(database=database) as session:
        for seed_name, seed_item in seed_entities:
            # 4b-i. 出边：seed → neighbor
            try:
                out_result = session.run(
                    """
                    MATCH (e:Entity {name: $name, item_name: $item_name})
                          -[r]->(neighbor:Entity {item_name: $item_name})
                    WHERE type(r) <> 'MENTIONED_IN'
                    RETURN e.name AS head, type(r) AS rel_type, neighbor.name AS tail
                    """,
                    name=seed_name, item_name=seed_item,
                )
                for rec in out_result:
                    head, rel, tail = rec["head"], rec["rel_type"], rec["tail"]
                    edge_key = (head, seed_item, rel, tail, seed_item)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        entity_relations.append({
                            "source": head,
                            "source_item": seed_item,
                            "relation": rel,
                            "target": tail,
                            "target_item": seed_item,
                        })
                        neighbor_entities.add((tail, seed_item))
            except Exception as e:
                logger.warning(f"一跳出边查询异常 (seed='{seed_name}' @ {seed_item}): {e}")

            # 4b-ii. 入边：neighbor → seed
            try:
                in_result = session.run(
                    """
                    MATCH (neighbor:Entity {item_name: $item_name})
                          -[r]->(e:Entity {name: $name, item_name: $item_name})
                    WHERE type(r) <> 'MENTIONED_IN'
                    RETURN neighbor.name AS head, type(r) AS rel_type, e.name AS tail
                    """,
                    name=seed_name, item_name=seed_item,
                )
                for rec in in_result:
                    head, rel, tail = rec["head"], rec["rel_type"], rec["tail"]
                    edge_key = (head, seed_item, rel, tail, seed_item)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        entity_relations.append({
                            "source": head,
                            "source_item": seed_item,
                            "relation": rel,
                            "target": tail,
                            "target_item": seed_item,
                        })
                        neighbor_entities.add((head, seed_item))
            except Exception as e:
                logger.warning(f"一跳入边查询异常 (seed='{seed_name}' @ {seed_item}): {e}")

    # 邻居节点排除种子节点自身（种子→种子 的边只计一次关系，不计为邻居）
    neighbor_entities -= seed_entities

    logger.info(
        f"KG 图搜索 - 4b 一跳扩展: {len(entity_relations)} 条边, "
        f"{len(neighbor_entities)} 个邻居节点"
    )

    # ---- 4c. MENTIONED_IN 反查 Chunk 加权打分 ----
    chunk_scores: Dict[str, int] = collections.defaultdict(int)
    chunk_mentions: Dict[str, int] = collections.defaultdict(int)

    with driver.session(database=database) as session:
        # 按 item_name 分组批量查询，减少 Cypher 调用次数
        for item_name in item_names:
            # 4c-i. 种子节点 MENTIONED_IN → Chunk（权重 2）
            seed_names_for_item = [
                n for n, i in seed_entities if i == item_name
            ]
            if seed_names_for_item:
                try:
                    seed_chunk_result = session.run(
                        """
                        MATCH (e:Entity {item_name: $item_name})
                              -[:MENTIONED_IN]->(c:Chunk)
                        WHERE e.name IN $names
                        RETURN c.id AS chunk_id
                        """,
                        item_name=item_name, names=seed_names_for_item,
                    )
                    for rec in seed_chunk_result:
                        cid = rec["chunk_id"]
                        chunk_scores[cid] += SEED_CHUNK_WEIGHT
                        chunk_mentions[cid] += 1
                except Exception as e:
                    logger.warning(f"种子 Chunk 反查异常 (item='{item_name}'): {e}")

            # 4c-ii. 邻居节点 MENTIONED_IN → Chunk（权重 1）
            neighbor_names_for_item = [
                n for n, i in neighbor_entities if i == item_name
            ]
            if neighbor_names_for_item:
                try:
                    neighbor_chunk_result = session.run(
                        """
                        MATCH (e:Entity {item_name: $item_name})
                              -[:MENTIONED_IN]->(c:Chunk)
                        WHERE e.name IN $names
                        RETURN c.id AS chunk_id
                        """,
                        item_name=item_name, names=neighbor_names_for_item,
                    )
                    for rec in neighbor_chunk_result:
                        cid = rec["chunk_id"]
                        chunk_scores[cid] += NEIGHBOR_CHUNK_WEIGHT
                        chunk_mentions[cid] += 1
                except Exception as e:
                    logger.warning(f"邻居 Chunk 反查异常 (item='{item_name}'): {e}")

    # 排序：得分降序 → 提及次数降序
    sorted_chunk_ids = sorted(
        chunk_scores.keys(),
        key=lambda cid: (chunk_scores[cid], chunk_mentions[cid]),
        reverse=True,
    )

    logger.info(
        f"KG 图搜索 - 4c Chunk 打分: {len(sorted_chunk_ids)} 个 Chunk "
        f"(top5 得分: {[chunk_scores[c] for c in sorted_chunk_ids[:5]]})"
    )

    # ---- 4d. Milvus 批量回填切片内容 + 按得分重排 ----
    kg_chunks: List[Dict] = []
    if sorted_chunk_ids:
        milvus_client = get_milvus_client()
        if milvus_client is not None:
            output_fields = [
                "chunk_id", "item_name", "content", "title",
                "parent_title", "part", "file_title",
            ]
            raw_chunks = fetch_chunks_by_chunk_ids(
                client=milvus_client,
                collection_name=milvus_config.chunks_collection,
                chunk_ids=sorted_chunk_ids,
                output_fields=output_fields,
                batch_size=100,
            )
            # 建立 chunk_id → 实体 的索引（key 统一为 int，Milvus INT64 主键）
            chunk_map: Dict[int, Dict] = {}
            for ch in raw_chunks:
                cid = ch.get("chunk_id")
                if cid is not None:
                    chunk_map[int(cid)] = ch

            # 按加权得分降序重排，构建 RRF 兼容格式
            # sorted_chunk_ids 来自 Neo4j 的 c.id 属性（导入侧以 str 存储），
            # 需转为 int 才能与 chunk_map 的 int key 匹配
            for cid_str in sorted_chunk_ids:
                try:
                    cid = int(cid_str)
                except (ValueError, TypeError):
                    logger.warning(f"KG 图搜索 - 4d: chunk_id '{cid_str}' 无法转为 int，跳过")
                    continue
                ch = chunk_map.get(cid)
                if ch is None:
                    logger.debug(
                        f"KG 图搜索 - 4d: chunk_id {cid}（Neo4j）在 Milvus 中未找到对应切片，可能数据已过期"
                    )
                    continue
                kg_chunks.append({
                    "id": cid,
                    "entity": {
                        "chunk_id": cid,
                        "item_name": ch.get("item_name", ""),
                        "content": ch.get("content", ""),
                        "title": ch.get("title", ""),
                        "parent_title": ch.get("parent_title", ""),
                        "part": ch.get("part", ""),
                        "file_title": ch.get("file_title", ""),
                    },
                    # 附加 KG 特有字段，供下游 answer_output 使用
                    "_kg_score": chunk_scores.get(cid_str, 0),
                    "_kg_mentions": chunk_mentions.get(cid_str, 0),
                })

            logger.info(
                f"KG 图搜索 - 4d Milvus 回填: {len(sorted_chunk_ids)} 个 ID "
                f"→ 命中 {len(kg_chunks)} 个切片"
            )
        else:
            logger.warning("Milvus 客户端不可用，跳过切片内容回填")

    # ---- 4e. 三元组转文本描述 ----
    kg_triple_text = _format_triple_text(entity_relations)

    logger.info(
        f"KG 图搜索完成: {len(kg_chunks)} 个切片（已排序）, "
        f"{len(entity_relations)} 条实体关系, "
        f"三元组文本 {len(kg_triple_text)} 字符"
    )
    return kg_chunks, entity_relations, kg_triple_text


def _format_triple_text(triples: List[Dict]) -> str:
    """将一跳三元组列表转为 LLM 易理解的文本描述，包含 item_name 以区分跨商品同名实体。"""
    if not triples:
        return ""
    lines = ["[知识图谱实体关系（一跳）]"]
    # 收集涉及的商品名，超过一个时显式展示 item_name 消歧
    all_items: Set[str] = set()
    for t in triples:
        all_items.add(t.get("source_item", ""))
        all_items.add(t.get("target_item", ""))
    multi_item = len(all_items) > 1

    for i, t in enumerate(triples, 1):
        src = t["source"]
        rel = t["relation"]
        dst = t["target"]
        if multi_item:
            si = t.get("source_item", "")
            ti = t.get("target_item", "")
            lines.append(f"{i}. \"{src}\"@{si} --[{rel}]--> \"{dst}\"@{ti}")
        else:
            lines.append(f"{i}. \"{src}\" --[{rel}]--> \"{dst}\"")
    return "\n".join(lines)


@node_log("node_search_kg")
def node_search_kg(state: QueryGraphState) -> dict:
    """
    节点功能：知识图谱检索（第4路并行检索路由）。

    利用 Neo4j 中的实体关系图补充向量检索的盲区——
    向量检索靠语义相似度，可能遗漏结构上相关但措辞不同的切片；
    KG 检索通过实体名精确匹配 + 关系遍历，能找到显式关联的切片。

    内部流程：
        1. 参数校验 + 剥离商品名
        2. LLM 提取查询中的关键实体术语（部件名、操作名等）
        3. Milvus 实体名向量对齐（BGE-M3 混合检索 → 映射到图谱标准实体名）
        4. Neo4j 图搜索：标准实体名 CONTAINS 匹配 → MENTIONED_IN → 切片 ID
           + 1-hop 关系扩展获取关联实体的切片
        5. Milvus 反查切片完整内容（通过 fetch_chunks_by_chunk_ids）
        6. 格式化为 RRF 兼容格式

    产出：
        - state["kg_chunks"]: KG 召回的切片列表（参与 RRF 融合）
        - state["kg_relations"]: 实体关系列表（后续 answer_output 可用）

    异常容错：
        - Neo4j 不可用 → 返回空结果，不阻断整体流程
        - LLM 提取失败 / 无关键词 → 回退为全量实体查询
        - Milvus 反查失败 → 返回空列表
    """
    # 任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # Step 1: 参数校验 + 剥离商品名
    item_names, cleaned_query = step_1_data_validates(state)

    # Step 2: LLM 提取关键实体术语（基于剥离商品名后的纯问题）
    raw_entities = step_2_extract_keywords(cleaned_query)

    # Step 3: Milvus 实体名向量对齐 → 映射到图谱中的标准实体名
    aligned_entities = step_3_align_entities_with_milvus(raw_entities, item_names)

    # Step 4: Neo4j 图搜索（种子定位 → 一跳扩展 → Chunk 打分 → Milvus 回填 → 三元组文本）
    kg_chunks, entity_relations, kg_triple_text = step_4_neo4j_graph_search(
        aligned_entities, item_names
    )

    # 任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info(
        f"KG检索完成: 输出 {len(kg_chunks)} 个切片（RRF融合用）, "
        f"{len(entity_relations)} 条实体关系（answer_output用）, "
        f"三元组文本 {len(kg_triple_text)} 字符"
    )

    return {
        "kg_chunks": kg_chunks,
        "kg_relations": entity_relations,
        "kg_triple_text": kg_triple_text,
    }


# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_kg 本地测试")
    print("=" * 50)

    # 模拟 state：产品"HAK 180 烫金机"的知识图谱已在 Neo4j 中
    mock_state: QueryGraphState = {
        "session_id": "test_kg_search_001",
        "original_query": "HAK 180 烫金机的漏电保护装置如何维护？",
        "rewritten_query": "HAK 180 烫金机的漏电保护装置如何维护和检测？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False,
    }

    try:
        result = node_search_kg(mock_state)
        kg_chunks = result.get("kg_chunks", [])
        kg_relations = result.get("kg_relations", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"KG Chunks 数量: {len(kg_chunks)}")
        for i, c in enumerate(kg_chunks[:5]):
            content_preview = str(c.get("entity", {}).get("content", ""))[:80]
            print(f"  [{i + 1}] chunk_id={c.get('id')}, content={content_preview}...")

        print(f"\nKG Relations 数量: {len(kg_relations)}")
        for i, r in enumerate(kg_relations[:10]):
            print(f"  [{i + 1}] ({r['source']})-[:{r['relation']}]->({r['target']})")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"node_search_kg 测试失败: {e}")
