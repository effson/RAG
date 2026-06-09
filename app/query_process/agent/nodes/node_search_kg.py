# KG（知识图谱）检索节点：利用 Neo4j 中的实体关系图，通过实体名匹配和图遍历定位相关切片
import sys
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.clients.neo4j_utils import get_neo4j_driver
from app.conf.neo4j_conf import get_config as get_neo4j_config
from app.clients.milvus_utils import get_milvus_client, fetch_chunks_by_chunk_ids
from app.conf.milvus_config import milvus_config
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# ---- 配置常量 ----
# KG 路在 RRF 融合时的权重（与其他路平权）
KG_WEIGHT = 1.0


@step_log("step_1_data_validates")
def step_1_data_validates(state: QueryGraphState):
    """
    获取参数并且校验
    :param state:
    :return: item_names, rewritten_query
    """
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if not item_names or not rewritten_query:
        logger.error("item_names或rewritten_query不存在,无法继续KG检索!")
        raise ValueError("item_names或rewritten_query不存在,无法继续KG检索!")
    return item_names, rewritten_query


@step_log("step_2_extract_keywords")
def step_2_extract_keywords(rewritten_query: str) -> List[str]:
    """
    调用 LLM 从用户改写问题中提取关键实体/部件/操作术语。
    这些关键词用于在 Neo4j 中做实体名 CONTAINS 匹配。

    注意：只提取具体的技术术语（部件名、操作名、参数名等），
    不提取设备名或泛泛的疑问词。

    :param rewritten_query: 改写后的问题
    :return: 关键词列表，提取失败返回空列表
    """
    llm = get_llm_client(json_mode=True)
    parser = JsonOutputParser()

    prompt = (
        f"从以下用户问题中提取关键实体/部件/操作术语（设备名和产品名除外）。\n"
        f"只提取具体的技术术语，如部件名、操作名、参数名、功能名等。最多提取 5 个。\n\n"
        f"用户问题: {rewritten_query}\n\n"
        f'请以 JSON 格式返回: {{"keywords": ["术语1", "术语2", ...]}}\n'
        f'如果提取不到任何术语，返回: {{"keywords": []}}'
    )

    messages = [
        SystemMessage(content="你是一个专业的技术术语提取助手，擅长从用户问题中识别关键实体和部件名称。"),
        HumanMessage(content=prompt),
    ]

    try:
        chain = llm | parser
        result = chain.invoke(messages)
        keywords = result.get("keywords", []) if isinstance(result, dict) else []
        logger.info(f"KG检索 - 提取到关键词: {keywords}")
        return keywords
    except Exception as e:
        logger.warning(f"KG检索 - 关键词提取失败: {e}，将使用空关键词列表回退")
        return []


@step_log("step_3_neo4j_graph_search")
def step_3_neo4j_graph_search(
    item_names: List[str], keywords: List[str]
) -> tuple:
    """
    在 Neo4j 中执行图搜索，通过实体名匹配和图关系遍历定位相关切片。

    查询策略（按优先级）：
    1. 关键词精确匹配实体名 → 沿 MENTIONED_IN 边获取直接提及的切片
    2. 1-hop 关系扩展 → 获取关联实体的切片（如问"漏电保护"也能找到"电源规格"切片）
    3. 无关键词回退 → 获取该产品下所有实体的切片（保证 KG 路始终有输出）

    :param item_names: 确认的产品名称列表
    :param keywords: LLM 提取的关键术语列表
    :return: (chunk_ids: List[str], entity_relations: List[Dict])
             chunk_ids 为 Neo4j 中 Chunk 节点的 id（字符串），
             entity_relations 为实体间关系列表 [{source, relation, target}]
    """
    driver = get_neo4j_driver()
    if driver is None:
        logger.warning("Neo4j 驱动不可用，跳过 KG 图搜索")
        return [], []

    neo4j_config = get_neo4j_config()
    database = neo4j_config.neo4j_database or "neo4j"

    chunk_ids: set = set()
    entity_relations: List[Dict] = []

    with driver.session(database=database) as session:
        for item_name in item_names:
            # ---- 策略1 + 策略2：基于关键词的实体匹配 + 1-hop 扩展 ----
            if keywords:
                for keyword in keywords:
                    # 1a. 直接实体 → 切片
                    try:
                        direct_result = session.run(
                            """
                            MATCH (e:Entity {item_name: $item_name})
                            WHERE e.name CONTAINS $keyword
                            MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                            RETURN DISTINCT c.id AS chunk_id
                            LIMIT 10
                            """,
                            item_name=item_name, keyword=keyword,
                        )
                        for record in direct_result:
                            chunk_ids.add(record["chunk_id"])
                    except Exception as e:
                        logger.warning(f"Neo4j 实体匹配查询异常 (keyword='{keyword}'): {e}")

                    # 1b. 1-hop 关联实体 → 切片
                    try:
                        related_result = session.run(
                            """
                            MATCH (e:Entity {item_name: $item_name})
                            WHERE e.name CONTAINS $keyword
                            MATCH (e)-[r]->(related:Entity {item_name: $item_name})
                            MATCH (related)-[:MENTIONED_IN]->(c:Chunk)
                            RETURN DISTINCT e.name AS source_entity,
                                   type(r) AS rel_type,
                                   related.name AS target_entity,
                                   c.id AS chunk_id
                            LIMIT 20
                            """,
                            item_name=item_name, keyword=keyword,
                        )
                        for record in related_result:
                            chunk_ids.add(record["chunk_id"])
                            entity_relations.append({
                                "source": record["source_entity"],
                                "relation": record["rel_type"],
                                "target": record["target_entity"],
                            })
                    except Exception as e:
                        logger.warning(f"Neo4j 关联实体查询异常 (keyword='{keyword}'): {e}")

            # ---- 策略3：无关键词时的全量回退 ----
            if not keywords:
                try:
                    fallback_result = session.run(
                        """
                        MATCH (e:Entity {item_name: $item_name})
                        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                        RETURN DISTINCT c.id AS chunk_id
                        LIMIT 30
                        """,
                        item_name=item_name,
                    )
                    for record in fallback_result:
                        chunk_ids.add(record["chunk_id"])
                except Exception as e:
                    logger.warning(f"Neo4j 全量回退查询异常: {e}")

                # 回退时也收集实体关系（给 answer_output 用）
                try:
                    rel_result = session.run(
                        """
                        MATCH (e1:Entity {item_name: $item_name})-[r]->(e2:Entity {item_name: $item_name})
                        RETURN DISTINCT e1.name AS source, type(r) AS rel_type, e2.name AS target
                        LIMIT 50
                        """,
                        item_name=item_name,
                    )
                    for record in rel_result:
                        entity_relations.append({
                            "source": record["source"],
                            "relation": record["rel_type"],
                            "target": record["target"],
                        })
                except Exception as e:
                    logger.warning(f"Neo4j 全量关系查询异常: {e}")

    logger.info(
        f"KG 图搜索完成: 找到 {len(chunk_ids)} 个关联切片, "
        f"{len(entity_relations)} 条实体关系"
    )
    return list(chunk_ids), entity_relations


@step_log("step_4_query_chunks_from_milvus")
def step_4_query_chunks_from_milvus(chunk_ids: List[str]) -> List[Dict]:
    """
    通过 chunk_id 列表从 Milvus 反查切片完整内容（文本、标题等）。
    使用 milvus_utils 中已有的 fetch_chunks_by_chunk_ids 工具函数。

    Neo4j 的 Chunk 节点只存 id，不含 content，所以必须回查 Milvus 补全。

    :param chunk_ids: Neo4j 返回的 chunk_id 字符串列表
    :return: Milvus 实体列表 [{chunk_id, content, title, ...}]
    """
    if not chunk_ids:
        return []

    milvus_client = get_milvus_client()
    if milvus_client is None:
        logger.warning("Milvus 客户端不可用，无法反查切片内容")
        return []

    output_fields = ["chunk_id", "item_name", "content", "title", "parent_title", "part", "file_title"]
    chunks = fetch_chunks_by_chunk_ids(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        chunk_ids=chunk_ids,
        output_fields=output_fields,
        batch_size=100,
    )

    logger.info(f"KG检索 - Milvus 反查: {len(chunk_ids)} 个 ID → 命中 {len(chunks)} 个切片")
    return chunks


@step_log("step_5_format_for_rrf")
def step_5_format_for_rrf(
    chunks: List[Dict], entity_relations: List[Dict]
) -> List[Dict]:
    """
    将 KG 检索结果格式化为 RRF 节点兼容的格式。

    RRF 节点期望的输入格式与 Milvus hybrid_search 返回一致：
    [{"id": chunk_id, "entity": {"chunk_id": ..., "content": ..., ...}}, ...]

    :param chunks: Milvus 反查返回的切片列表
    :param entity_relations: 实体关系列表（传给 answer_output 用，此处仅透传）
    :return: RRF 兼容的切片列表
    """
    kg_chunks = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        kg_chunks.append({
            "id": chunk_id,
            "entity": {
                "chunk_id": chunk_id,
                "item_name": chunk.get("item_name", ""),
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "parent_title": chunk.get("parent_title", ""),
                "part": chunk.get("part", ""),
                "file_title": chunk.get("file_title", ""),
            },
        })
    return kg_chunks


@node_log("node_search_kg")
def node_search_kg(state: QueryGraphState) -> dict:
    """
    节点功能：知识图谱检索（第4路并行检索路由）。

    利用 Neo4j 中的实体关系图补充向量检索的盲区——
    向量检索靠语义相似度，可能遗漏结构上相关但措辞不同的切片；
    KG 检索通过实体名精确匹配 + 关系遍历，能找到显式关联的切片。

    内部流程：
        1. 参数校验（item_names / rewritten_query）
        2. LLM 提取查询中的关键实体术语（部件名、操作名等）
        3. Neo4j 图搜索：实体 CONTAINS 匹配 → MENTIONED_IN → 切片 ID
           + 1-hop 关系扩展获取关联实体的切片
        4. Milvus 反查切片完整内容（通过 fetch_chunks_by_chunk_ids）
        5. 格式化为 RRF 兼容格式

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

    # Step 1: 参数校验
    item_names, rewritten_query = step_1_data_validates(state)

    # Step 2: LLM 提取关键实体术语
    keywords = step_2_extract_keywords(rewritten_query)

    # Step 3: Neo4j 图搜索 → chunk_id 列表 + 实体关系
    chunk_ids, entity_relations = step_3_neo4j_graph_search(item_names, keywords)

    # Step 4: Milvus 反查切片内容
    chunks = step_4_query_chunks_from_milvus(chunk_ids)

    # Step 5: 格式化为 RRF 兼容
    kg_chunks = step_5_format_for_rrf(chunks, entity_relations)

    # 任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info(
        f"KG检索完成: 输出 {len(kg_chunks)} 个切片（RRF融合用）, "
        f"{len(entity_relations)} 条实体关系（answer_output用）"
    )

    return {
        "kg_chunks": kg_chunks,
        "kg_relations": entity_relations,
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
