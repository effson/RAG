"""
================================================================================
RAGAS RAG 评估程序 (RAGAS-based RAG Evaluation Script)
================================================================================

功能说明:
    基于 RAGAS (Retrieval Augmented Generation Assessment) 框架对 RAG 系统进行
    全维度自动化评估。评估流程为:
        1. 读取 QA 数据集 (eval/qa.csv)
        2. 对每个问题运行完整的 RAG 查询管线，收集生成答案和检索上下文
        3. 使用 RAGAS 框架计算 5 项核心评估指标
        4. 将评估结果保存到 eval/qa_result.csv

评估指标:
    - Faithfulness    (忠实度/幻觉度):  评估生成答案是否忠实于检索上下文，检测幻觉
    - Answer Relevance (答案相关性):     评估答案与问题的语义相关程度
    - Context Precision (上下文精确度):  评估检索到的文档排序质量（相关文档是否排前）
    - Context Recall   (上下文召回率):   评估检索上下文是否覆盖参考答案中的关键信息
    - Answer Correctness (答案正确性):   评估答案相对于参考答案的语义正确性和事实准确性

输入:
    eval/qa.csv — 包含 "question" (问题) 和 "ground_truth" (参考答案) 两列

输出:
    eval/qa_result.csv — 包含以下列:
        question, answer, context,
        Faithfulness, Answer Relevance, Context Precision,
        Context Recall, Answer Correctness

依赖:
    - 项目自有的 LLM 客户端 (app.lm.lm_utils.get_llm_client)
    - 项目自有的 BGE-M3 嵌入模型 (app.lm.embedding_utils.get_bge_m3_ef)
    - RAGAS 框架 (需额外安装: uv pip install ragas)

注意事项:
    - 不要修改项目原有的代码
    - 运行前需确保 .env 中已正确配置 LLM 和嵌入模型参数
    - 运行前需确保 Milvus / Neo4j / MongoDB 等服务已启动（如需完整管线）

作者: 基石智库团队
日期: 2026-06-11
================================================================================
"""

import sys
import json
import time
import uuid
import warnings
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

# ==============================================================================
# Monkey-patch: 修复 ragas 与新版 langchain-community 的兼容性问题
# ==============================================================================
# ragas 在其 llms/base.py 中硬编码了:
#   from langchain_community.chat_models.vertexai import ChatVertexAI
#   from langchain_community.llms import VertexAI
# 但新版 langchain-community (>0.3) 已将 ChatVertexAI/VertexAI 迁移到独立包
# langchain-google-vertexai 中。
# 由于项目不使用 VertexAI，此补丁注入哑元类以绕过导入错误。
# 此补丁必须在任何 ragas 导入之前执行。

import importlib

def _patch_langchain_community_for_ragas():
    """
    为 langchain_community 注入缺失的 VertexAI 模块/类，
    确保 ragas 可以正常导入而不报 ModuleNotFoundError。
    仅在 langchain_community 中不存在 vertexai 子模块时生效。
    """
    try:
        from langchain_community.chat_models import vertexai
        # 已存在，无需补丁
        return
    except ImportError:
        pass

    # 创建哑元 ChatVertexAI 类
    class _DummyChatVertexAI:
        """哑元 ChatVertexAI 占位类 —— 仅用于绕过 ragas 的导入检查"""
        pass

    class _DummyVertexAI:
        """哑元 VertexAI 占位类 —— 仅用于绕过 ragas 的导入检查"""
        pass

    # 在 langchain_community 中注入 vertexai 模块
    import langchain_community.chat_models as chat_models_pkg
    import types

    # 创建 chat_models.vertexai 子模块
    vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_module.ChatVertexAI = _DummyChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
    setattr(chat_models_pkg, "vertexai", vertexai_module)

    # 补丁 VertexAI: 新版 langchain_community.llms 可能也没有 VertexAI
    try:
        from langchain_community.llms import VertexAI
    except ImportError:
        import langchain_community.llms as llms_pkg
        if not hasattr(llms_pkg, "VertexAI"):
            llms_pkg.VertexAI = _DummyVertexAI  # type: ignore

_patch_langchain_community_for_ragas()

# ==============================================================================
# 路径与项目环境初始化
# ==============================================================================

# 获取项目根目录（eval 的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 将项目根目录加入到 sys.path，确保可以导入 app.* 模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 环境变量（必须在导入 app.* 之前加载）
load_dotenv(PROJECT_ROOT / ".env")

# ==============================================================================
# 项目内部依赖（延迟导入，确保环境变量已加载）
# ==============================================================================

from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state, QueryGraphState
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import get_bge_m3_ef
from app.conf.lm_config import lm_config
from app.conf.embedding_config import embedding_config
from app.core.logger import logger as app_logger

# ==============================================================================
# 日志配置
# ==============================================================================

# 使用 loguru 为评估脚本创建独立的 logger（避免与项目 logger 冲突）
from loguru import logger as eval_logger

# 移除默认 handler，添加自定义格式
eval_logger.remove()
eval_logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

# ==============================================================================
# RAGAS 依赖导入与兼容性检查
# ==============================================================================

# 抑制 RAGAS evaluate() / 旧指标路径 的弃用警告（功能正常）
warnings.filterwarnings(
    "ignore",
    message=".*evaluate.* is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Importing.*from.*ragas.metrics.*",
    category=DeprecationWarning,
)

try:
    from openai import OpenAI as OpenAIClient

    from ragas import evaluate
    # 使用旧路径的预构建 Metric 实例（兼容 evaluate() API）
    from ragas.metrics import (
        faithfulness as ragas_faithfulness,
        answer_relevancy as ragas_answer_relevancy,
        context_precision as ragas_context_precision,
        context_recall as ragas_context_recall,
        answer_correctness as ragas_answer_correctness,
    )
    # RAGAS 0.4.x 现代 API: llm_factory + BaseRagasEmbedding
    from ragas.llms import llm_factory
    from ragas.embeddings import BaseRagasEmbedding
    from ragas.run_config import RunConfig
    from datasets import Dataset as HFDataset
    RAGAS_AVAILABLE = True
    eval_logger.info("RAGAS 框架导入成功")
except ImportError as e:
    RAGAS_AVAILABLE = False
    eval_logger.warning(
        f"RAGAS 未安装 ({e})。请运行: uv pip install ragas datasets openai"
    )

# ==============================================================================
# 常量定义
# ==============================================================================

# 评估用 CSV 文件路径
QA_CSV_PATH: Path = PROJECT_ROOT / "eval" / "qa.csv"
RESULT_CSV_PATH: Path = PROJECT_ROOT / "eval" / "qa_result.csv"

# 评估超时设置
QUERY_TIMEOUT_SECONDS: int = 120  # 单次查询管线最大等待时间（秒）
RAGAS_LLM_TIMEOUT_SECONDS: int = 60  # RAGAS 评估中单次 LLM 调用超时

# 评估指标中文名称映射（用于日志输出）
METRIC_CN_MAP: Dict[str, str] = {
    "faithfulness": "忠实度/幻觉度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
    "answer_correctness": "答案正确性",
}


# ==============================================================================
# BGE-M3 嵌入模型 RAGAS 包装器
# ==============================================================================

class BGEMM3RagasEmbedding(BaseRagasEmbedding):
    """
    将项目自有的 BGEM3EmbeddingFunction (PyMilvus 接口) 包装为 RAGAS 0.4.x
    兼容的 BaseRagasEmbedding 实现。

    同时提供旧版 LangChain Embeddings 接口 (embed_query / embed_documents) 以
    兼容 ragas.metrics 中仍在使用 LangChain 接口的旧指标（如 answer_relevancy）。

    本包装器复用了项目已加载的 BGE-M3 模型（单例），避免重复加载。

    Attributes:
        _ef: 项目全局单例的 BGEM3EmbeddingFunction 实例（延迟加载）

    Example:
        >>> embedding = BGEMM3RagasEmbedding()
        >>> vec = embedding.embed_text("查询文本")           # RAGAS 新接口
        >>> vec = embedding.embed_query("查询文本")          # LangChain 旧接口
        >>> vecs = embedding.embed_texts(["文本1", "文本2"]) # 批量编码
    """

    def __init__(self, cache=None) -> None:
        """
        初始化包装器

        Args:
            cache: 可选的 RAGAS CacheInterface，用于缓存嵌入结果
        """
        super().__init__(cache=cache)
        self._ef = None  # 延迟加载，避免在导入时触发模型加载

    def _get_ef(self):
        """
        获取 BGEM3EmbeddingFunction 实例（延迟加载/单例复用）

        Returns:
            BGEM3EmbeddingFunction 实例

        Raises:
            RuntimeError: 模型初始化失败
        """
        if self._ef is None:
            try:
                self._ef = get_bge_m3_ef()
                eval_logger.info("BGE-M3 嵌入模型已加载（供 RAGAS 评估使用）")
            except Exception as e:
                raise RuntimeError(f"BGE-M3 模型加载失败: {e}") from e
        return self._ef

    # ── RAGAS 现代 API (BaseRagasEmbedding 抽象方法) ──

    def embed_text(self, text: str, **kwargs) -> List[float]:
        """
        为单条文本生成稠密向量嵌入（RAGAS 新接口，同步）

        Args:
            text: 输入文本
            **kwargs: 额外参数（保留兼容）

        Returns:
            稠密向量 (float 列表)，维度 = 1024

        Raises:
            ValueError: text 为空
            RuntimeError: 向量生成失败
        """
        if not text or not isinstance(text, str):
            raise ValueError("text 必须为非空字符串")
        try:
            ef = self._get_ef()
            result = ef.encode_queries([text])
            return result["dense"][0].tolist()
        except Exception as e:
            raise RuntimeError(f"单条文本向量生成失败: {e}") from e

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        """
        为单条文本生成稠密向量嵌入（RAGAS 新接口，异步）
        """
        import asyncio
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        为多条文本批量生成稠密向量嵌入（覆写基类以使用原生批量编码）

        Args:
            texts: 输入文本列表
            **kwargs: 额外参数

        Returns:
            稠密向量列表

        Raises:
            ValueError: texts 为空或不是列表
            RuntimeError: 向量生成失败
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("texts 必须为非空的文本列表")
        try:
            ef = self._get_ef()
            result = ef.encode_documents(texts)
            return [emb.tolist() for emb in result["dense"]]
        except Exception as e:
            raise RuntimeError(f"批量文本向量生成失败: {e}") from e

    async def aembed_texts(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """异步批量嵌入"""
        import asyncio
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    # ── 旧版 LangChain Embeddings 接口（兼容 ragas.metrics 中旧指标的 embed_query 调用）──

    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成嵌入（LangChain 旧接口，委托到 embed_text）

        ragas.metrics 中的 answer_relevancy / answer_correctness 等旧指标
        内部仍通过 embed_query() 调用嵌入模型，提供此方法以兼容。
        """
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入（LangChain 旧接口，委托到 embed_texts）
        """
        return self.embed_texts(texts)


# ==============================================================================
# QA 数据加载与校验
# ==============================================================================

def load_qa_data(csv_path: Path) -> pd.DataFrame:
    """
    从 CSV 文件中加载 QA 数据集并进行校验

    预期 CSV 格式:
        - question:    用户提出的问题（必需）
        - ground_truth: 参考答案 / 标准答案（必需）

    Args:
        csv_path: QA CSV 文件路径

    Returns:
        包含 question 和 ground_truth 列的 pandas DataFrame

    Raises:
        FileNotFoundError: CSV 文件不存在
        ValueError: CSV 文件缺少必需的列或数据为空
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"QA 数据集文件不存在: {csv_path}\n"
            f"请在 {csv_path.parent} 目录下创建 qa.csv 文件，"
            f"包含 'question' 和 'ground_truth' 两列。"
        )

    eval_logger.info(f"正在读取 QA 数据集: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        # 尝试 GBK 编码（兼容 Windows 中文环境）
        df = pd.read_csv(csv_path, encoding="gbk")

    # 校验必需列
    required_columns = {"question", "ground_truth"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV 文件缺少必需的列: {missing_columns}\n"
            f"当前列: {list(df.columns)}\n"
            f"请确保 CSV 包含 'question' 和 'ground_truth' 两列。"
        )

    # 清洗数据：删除 question 或 ground_truth 为空的行
    original_count = len(df)
    df = df.dropna(subset=["question", "ground_truth"])
    df = df[df["question"].str.strip() != ""]
    df = df[df["ground_truth"].str.strip() != ""]

    if len(df) == 0:
        raise ValueError("QA 数据集中没有有效的数据行（question 和 ground_truth 均为空）")

    if len(df) < original_count:
        eval_logger.warning(
            f"已过滤 {original_count - len(df)} 行无效数据（question 或 ground_truth 为空）"
        )

    eval_logger.info(f"成功加载 {len(df)} 条 QA 数据")
    return df


# ==============================================================================
# RAG 管线执行
# ==============================================================================

def run_rag_query(
    question: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对单个问题执行完整的 RAG 查询管线 (query_app.invoke)

    管线流程:
        node_item_name_confirm → [node_search_embedding,
                                   node_search_embedding_hyde,
                                   node_search_kg,
                                   node_web_search_mcp]
                              → node_rrf → node_rerank → node_answer_output

    Args:
        question: 用户提出的原始问题
        session_id: 会话标识符（可选，不提供则自动生成 UUID）

    Returns:
        包含以下字段的字典:
            - success: 是否执行成功 (True/False)
            - answer:  生成的答案文本（失败时为空字符串）
            - reranked_docs: 重排序后的检索上下文列表
            - error:   错误信息（成功时为 None）
            - state:   完整的管线最终状态（用于调试，出错时为 None）

    Raises:
        不抛出异常，所有错误被捕获并返回在字典中
    """
    if session_id is None:
        session_id = f"eval_{uuid.uuid4().hex[:12]}"

    eval_logger.debug(f"开始处理问题 [{session_id}]: {question[:80]}...")

    # 构造管线初始状态（非流式模式，适合评估场景）
    initial_state = create_query_default_state(
        session_id=session_id,
        original_query=question,
        is_stream=False,
    )

    try:
        # 执行完整的 LangGraph 查询管线
        # query_app.invoke() 会依次执行所有节点并返回最终状态
        final_state = query_app.invoke(initial_state)

        # 提取关键字段
        answer = final_state.get("answer", "") or ""
        reranked_docs = final_state.get("reranked_docs", []) or []

        # 记录答案长度和上下文数量
        eval_logger.debug(
            f"[{session_id}] 管线完成: answer_len={len(answer)}, "
            f"reranked_docs_count={len(reranked_docs)}"
        )

        # 检查是否得到了有意义的答案
        if not answer:
            eval_logger.warning(f"[{session_id}] 管线返回的答案为空")
        if not reranked_docs:
            eval_logger.warning(f"[{session_id}] 管线返回的 reranked_docs 为空")

        return {
            "success": True,
            "answer": answer,
            "reranked_docs": reranked_docs,
            "error": None,
            "state": final_state,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        eval_logger.error(
            f"[{session_id}] 查询管线执行失败:\n"
            f"  问题: {question[:100]}\n"
            f"  错误: {error_msg}\n"
            f"  堆栈: {traceback.format_exc()}"
        )
        return {
            "success": False,
            "answer": "",
            "reranked_docs": [],
            "error": error_msg,
            "state": None,
        }


# ==============================================================================
# 上下文格式化
# ==============================================================================

def format_reranked_docs_for_csv(reranked_docs: List[Dict]) -> str:
    """
    将 reranked_docs 列表格式化为 CSV 友好的字符串

    每个文档包含 title, text, url, type, score 字段。
    输出以 JSON 序列化形式保存完整信息，便于后续分析。

    Args:
        reranked_docs: reranker 排序后的文档列表

    Returns:
        JSON 字符串表示的上下文信息
    """
    if not reranked_docs:
        return "[]"

    try:
        # 对文本字段做截断处理，避免单行过长
        compact_docs = []
        for doc in reranked_docs:
            text = doc.get("text", "")
            if len(text) > 500:
                text = text[:500] + "..."
            compact_docs.append({
                "title": doc.get("title", ""),
                "text": text,
                "url": doc.get("url"),
                "type": doc.get("type", ""),
                "score": doc.get("score", 0.0),
            })
        return json.dumps(compact_docs, ensure_ascii=False)
    except Exception:
        # JSON 序列化失败时的回退方案
        texts = [doc.get("text", "")[:200] for doc in reranked_docs]
        return " | ".join(texts)


def extract_context_texts(reranked_docs: List[Dict]) -> List[str]:
    """
    从 reranked_docs 中提取纯文本上下文列表，供 RAGAS 评估使用

    RAGAS 的 contexts 参数期望一个 List[str]，每条为一个检索到的文档/片段。
    本函数从 reranked_docs 的每个 dict 中提取 "text" 字段。

    Args:
        reranked_docs: reranker 排序后的文档列表 [{title, text, url, type, score}, ...]

    Returns:
        文本内容列表 [str, str, ...]
    """
    if not reranked_docs:
        return []
    return [doc.get("text", "") for doc in reranked_docs if doc.get("text")]


# ==============================================================================
# RAGAS 评估器构建
# ==============================================================================

def build_ragas_evaluator() -> Tuple[Any, Any, List[Any]]:
    """
    构建 RAGAS 评估器，包括 LLM、嵌入模型和评估指标列表

    RAGAS 0.4.x 适配方案:
        - LLM:  通过 llm_factory() 创建 InstructorLLM，使用项目的 DashScope
                兼容端点（OpenAI client 指向 OPENAI_BASE_URL）
        - Embed: 通过自定义 BGEMM3RagasEmbedding 复用项目已加载的 BGE-M3，
                同时提供 BaseRagasEmbedding 现代接口和 embed_query 旧接口
        - Metrics: 使用 ragas.metrics 旧路径的预构建 Metric 实例（兼容
                  evaluate() API），手动注入 LLM/embeddings

    Returns:
        (evaluator_llm, evaluator_embeddings, metrics) 三元组:
            - evaluator_llm: RAGAS InstructorLLM 实例
            - evaluator_embeddings: BGEMM3RagasEmbedding 实例
            - metrics: RAGAS 评估指标对象列表 (已注入 LLM/Embeddings)

    Raises:
        ImportError: RAGAS 未安装
        RuntimeError: LLM 或嵌入模型初始化失败
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS 框架未安装，请先运行: uv pip install ragas datasets openai"
        )

    eval_logger.info("正在构建 RAGAS 评估器...")

    # ── 1. 创建 LLM 评估器 ──
    # RAGAS 0.4.x 要求使用 llm_factory 创建 InstructorLLM，
    # 不支持旧的 LangchainLLMWrapper
    # 使用项目 .env 中配置的 DashScope 兼容端点（OpenAI 兼容协议）
    try:
        openai_client = OpenAIClient(
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            timeout=300.0,  # 评估 LLM 调用较长，放宽超时避免 TimeoutError
        )
        model_name = lm_config.llm_model or "qwen-flash"
        evaluator_llm = llm_factory(
            model=model_name,
            provider="openai",
            client=openai_client,
        )
        eval_logger.info(f"RAGAS LLM 评估器已就绪: model={model_name}")
    except Exception as e:
        raise RuntimeError(f"RAGAS LLM 评估器初始化失败: {e}") from e

    # ── 2. 创建 Embedding 评估器 ──
    # 使用自定义 BGEMM3RagasEmbedding，复用项目已加载的 BGE-M3 模型
    try:
        evaluator_embeddings = BGEMM3RagasEmbedding()
        eval_logger.info("RAGAS Embedding 评估器已就绪: model=BGE-M3 (local)")
    except Exception as e:
        raise RuntimeError(f"RAGAS Embedding 评估器初始化失败: {e}") from e

    # ── 3. 创建评估指标列表 ──
    # 5 项核心指标，使用 ragas.metrics 中预构建的 Metric 实例
    # 这些旧路径指标兼容 evaluate() API 且是 Metric 的子类
    # 需要手动注入 LLM / embeddings 到各个 Metric 实例上
    ragas_faithfulness.llm = evaluator_llm
    ragas_answer_relevancy.llm = evaluator_llm
    ragas_answer_relevancy.embeddings = evaluator_embeddings
    ragas_context_precision.llm = evaluator_llm
    ragas_context_recall.llm = evaluator_llm
    ragas_answer_correctness.llm = evaluator_llm
    ragas_answer_correctness.embeddings = evaluator_embeddings

    metrics = [
        ragas_faithfulness,
        ragas_answer_relevancy,
        ragas_context_precision,
        ragas_context_recall,
        ragas_answer_correctness,
    ]
    eval_logger.info(
        f"RAGAS 评估指标已就绪: {[m.name for m in metrics]}"
    )

    return evaluator_llm, evaluator_embeddings, metrics


# ==============================================================================
# RAGAS 评估执行
# ==============================================================================

def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    metrics: List[Any],
) -> pd.DataFrame:
    """
    使用 RAGAS 框架对 RAG 结果进行批量评估

    Args:
        questions:       问题列表 [str, ...]
        answers:         生成的答案列表 [str, ...]
        contexts_list:   检索上下文列表 [[str, str, ...], ...]
                         （每条问题对应一个字符串列表）
        ground_truths:   参考答案列表 [str, ...]
        evaluator_llm:   RAGAS 兼容的 LLM 评估器
        evaluator_embeddings: RAGAS 兼容的嵌入模型评估器
        metrics:         评估指标对象列表

    Returns:
        包含所有评估指标得分的 pandas DataFrame

    Raises:
        ValueError: 数据长度不一致或数据为空
        RuntimeError: RAGAS 评估过程失败
    """
    # ── 数据校验 ──
    total = len(questions)
    if total == 0:
        raise ValueError("没有可评估的数据（所有管线调用均失败）")

    # 确保所有列表长度一致
    for name, lst in [
        ("answers", answers),
        ("contexts_list", contexts_list),
        ("ground_truths", ground_truths),
    ]:
        if len(lst) != total:
            raise ValueError(
                f"数据长度不一致: questions={total}, {name}={len(lst)}"
            )

    eval_logger.info(f"开始 RAGAS 评估，共 {total} 条数据...")

    # ── 构建 HuggingFace Dataset ──
    # RAGAS 使用 HuggingFace Dataset 作为数据输入格式
    eval_dataset = HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    eval_logger.debug(
        f"评估数据集已构建: {len(eval_dataset)} 条记录\n"
        f"  平均 contexts 数量: "
        f"{sum(len(c) for c in contexts_list) / max(len(contexts_list), 1):.1f}"
    )

    # ── 执行 RAGAS 评估 ──
    eval_logger.info("正在执行 RAGAS 评估（可能需要数分钟，视数据量和模型速度而定）...")
    start_time = time.time()

    # RAGAS 默认 max_workers=16 会对 API 造成过大并发压力，
    # 导致 DashScope/Qwen 超时。限制为 3 个并发 worker，
    # timeout 从默认 180s 提高到 300s 以适应长文本评估
    run_config = RunConfig(
        max_workers=3,
        timeout=300,
        max_retries=3,
    )

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
        )
    except Exception as e:
        raise RuntimeError(f"RAGAS 评估执行失败: {e}") from e

    elapsed = time.time() - start_time
    eval_logger.info(f"RAGAS 评估完成，耗时 {elapsed:.1f} 秒")

    # ── 转换结果为 DataFrame ──
    result_df = result.to_pandas()

    # ── 打印评估摘要 ──
    _print_evaluation_summary(result_df)

    return result_df


def _print_evaluation_summary(result_df: pd.DataFrame) -> None:
    """
    打印评估结果的统计摘要到日志

    Args:
        result_df: RAGAS 评估结果 DataFrame
    """
    eval_logger.info("=" * 60)
    eval_logger.info("RAGAS 评估摘要")
    eval_logger.info("=" * 60)

    # 所有可能的评估指标列名
    metric_columns = [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "answer_correctness",
    ]

    for col in metric_columns:
        if col in result_df.columns:
            values = result_df[col].dropna()
            if len(values) > 0:
                cn_name = METRIC_CN_MAP.get(col, col)
                eval_logger.info(
                    f"  {cn_name} ({col}): "
                    f"mean={values.mean():.4f}, "
                    f"median={values.median():.4f}, "
                    f"min={values.min():.4f}, "
                    f"max={values.max():.4f}"
                )

    eval_logger.info("=" * 60)


# ==============================================================================
# 结果保存
# ==============================================================================

def save_evaluation_results(
    output_path: Path,
    questions: List[str],
    answers: List[str],
    contexts_formatted: List[str],
    metrics_df: pd.DataFrame,
) -> None:
    """
    将评估结果保存到 CSV 文件

    输出 CSV 包含以下列:
        - question:              原始问题
        - answer:                生成的答案
        - context:               重排序后的上下文（JSON 格式字符串）
        - Faithfulness:          忠实度/幻觉度
        - Answer Relevance:      答案相关性
        - Context Precision:     上下文精确度
        - Context Recall:        上下文召回率
        - Answer Correctness:    答案正确性

    Args:
        output_path:          输出 CSV 文件路径
        questions:            原始问题列表
        answers:              RAG 生成的答案列表
        contexts_formatted:   格式化的上下文列表（JSON 字符串）
        metrics_df:           RAGAS 评估指标 DataFrame

    Raises:
        ValueError: 数据长度不一致
    """
    total = len(questions)
    if not (len(answers) == len(contexts_formatted) == len(metrics_df) == total):
        raise ValueError(
            f"数据长度不一致: questions={total}, answers={len(answers)}, "
            f"contexts={len(contexts_formatted)}, metrics={len(metrics_df)}"
        )

    # ── 构建输出 DataFrame ──
    output_df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "context": contexts_formatted,
    })

    # 合并 RAGAS 评估指标列（列名映射为可读名称）
    column_mapping = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "answer_correctness": "Answer Correctness",
    }

    for ragas_col, display_col in column_mapping.items():
        if ragas_col in metrics_df.columns:
            output_df[display_col] = metrics_df[ragas_col].values
        else:
            eval_logger.warning(f"评估指标 '{ragas_col}' 在 RAGAS 结果中不存在，将填充为空")

    # ── 保存到 CSV ──
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    eval_logger.info(f"评估结果已保存至: {output_path}")
    eval_logger.info(f"共 {len(output_df)} 条记录，{len(output_df.columns)} 个字段")


# ==============================================================================
# 主流程
# ==============================================================================

def main() -> None:
    """
    评估主流程

    步骤:
        1. 检查 RAGAS 是否可用
        2. 加载 QA 数据集
        3. 构建 RAGAS 评估器（LLM + Embedding + Metrics）
        4. 逐条运行 RAG 管线，收集 answers 和 contexts
        5. 对有有效结果的数据执行 RAGAS 评估
        6. 合并所有结果并保存到 CSV
    """
    eval_logger.info("=" * 60)
    eval_logger.info("基石智库 RAGAS 评估程序 启动")
    eval_logger.info(f"项目根目录: {PROJECT_ROOT}")
    eval_logger.info(f"QA 数据路径: {QA_CSV_PATH}")
    eval_logger.info(f"结果输出路径: {RESULT_CSV_PATH}")
    eval_logger.info("=" * 60)

    # ── 步骤 0: 检查 RAGAS 可用性 ──
    if not RAGAS_AVAILABLE:
        eval_logger.error(
            "RAGAS 框架未安装，无法继续。\n"
            "请运行以下命令安装:\n"
            "  uv pip install ragas datasets\n"
            "或:\n"
            "  pip install ragas datasets"
        )
        return

    # ── 步骤 1: 加载 QA 数据集 ──
    try:
        df = load_qa_data(QA_CSV_PATH)
    except (FileNotFoundError, ValueError) as e:
        eval_logger.error(f"加载 QA 数据集失败: {e}")
        return

    # ── 步骤 2: 构建 RAGAS 评估器 ──
    try:
        evaluator_llm, evaluator_embeddings, metrics = build_ragas_evaluator()
    except (ImportError, RuntimeError) as e:
        eval_logger.error(f"构建 RAGAS 评估器失败: {e}")
        return

    # ── 步骤 3: 逐条运行 RAG 管线 ──
    questions: List[str] = []
    ground_truths: List[str] = []
    answers: List[str] = []
    contexts_formatted: List[str] = []
    contexts_for_ragas: List[List[str]] = []

    success_count = 0
    fail_count = 0

    eval_logger.info(f"开始逐条处理 {len(df)} 个问题...")

    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip()
        current_idx = idx + 1 if isinstance(idx, int) else success_count + fail_count + 1

        eval_logger.info(
            f"[{current_idx}/{len(df)}] 处理问题: {question[:80]}{'...' if len(question) > 80 else ''}"
        )

        # 执行 RAG 管线
        result = run_rag_query(question)

        if result["success"]:
            answer = result["answer"]
            reranked_docs = result["reranked_docs"]

            # 即使 answer 或 reranked_docs 为空也保留（后续评估会反映这些问题）
            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append(answer)
            contexts_formatted.append(format_reranked_docs_for_csv(reranked_docs))
            contexts_for_ragas.append(extract_context_texts(reranked_docs))
            success_count += 1

            eval_logger.info(
                f"  ✓ 成功 | answer_len={len(answer)}, "
                f"context_chunks={len(reranked_docs)}"
            )
        else:
            # 管线执行失败 — 仍然记录问题，但 answer 和 context 为空
            questions.append(question)
            ground_truths.append(ground_truth)
            answers.append("")
            contexts_formatted.append("[]")
            contexts_for_ragas.append([])
            fail_count += 1

            eval_logger.warning(
                f"  ✗ 管线执行失败（错误已记录）: {result['error']}"
            )

        # 添加短暂延迟，避免对服务端造成过大压力
        time.sleep(0.5)

    eval_logger.info(
        f"管线执行完毕: 成功 {success_count}/{len(df)}, 失败 {fail_count}/{len(df)}"
    )

    if success_count == 0:
        eval_logger.error("所有管线调用均失败，无法进行 RAGAS 评估。请检查服务是否正常启动。")
        return

    # ── 步骤 4: 执行 RAGAS 评估 ──
    # 筛选有有效答案和上下文的数据进行评估
    valid_indices = [
        i for i in range(len(questions))
        if answers[i] and contexts_for_ragas[i]
    ]

    if not valid_indices:
        eval_logger.error("没有同时具备有效答案和上下文的记录，无法进行 RAGAS 评估。")
        # 即使无法评估，也保存一份基础结果供检查
        save_basic_results(questions, answers, contexts_formatted)
        return

    eval_logger.info(
        f"共 {len(valid_indices)}/{len(questions)} 条记录可用于 RAGAS 评估"
        f"（筛选条件: answer 和 context 均非空）"
    )

    # 筛选有效数据
    eval_questions = [questions[i] for i in valid_indices]
    eval_answers = [answers[i] for i in valid_indices]
    eval_contexts = [contexts_for_ragas[i] for i in valid_indices]
    eval_ground_truths = [ground_truths[i] for i in valid_indices]

    try:
        metrics_df = run_ragas_evaluation(
            questions=eval_questions,
            answers=eval_answers,
            contexts_list=eval_contexts,
            ground_truths=eval_ground_truths,
            evaluator_llm=evaluator_llm,
            evaluator_embeddings=evaluator_embeddings,
            metrics=metrics,
        )
    except Exception as e:
        eval_logger.error(f"RAGAS 评估执行失败: {e}")
        # 即使评估失败也保存基础结果
        save_basic_results(questions, answers, contexts_formatted)
        return

    # ── 步骤 5: 合并并保存最终结果 ──
    # metrics_df 仅包含有效评估行的指标，
    # 我们需要将所有行（包括未评估的）合并到最终输出中

    final_questions = questions
    final_answers = answers
    final_contexts = contexts_formatted

    # 创建包含所有行的完整 DataFrame
    all_results_df = pd.DataFrame({
        "question": final_questions,
        "answer": final_answers,
        "context": final_contexts,
    })

    # 为所有行添加评估指标列（默认 NaN）
    for col in ["faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness"]:
        all_results_df[col] = float("nan")

    # 将有效行的评估指标填充进去
    # metrics_df 的行顺序对应 valid_indices 的顺序
    for metric_col in metrics_df.columns:
        if metric_col in all_results_df.columns:
            for local_idx, global_idx in enumerate(valid_indices):
                all_results_df.at[global_idx, metric_col] = metrics_df.iloc[local_idx][metric_col]

    # 保存
    save_evaluation_results(
        output_path=RESULT_CSV_PATH,
        questions=final_questions,
        answers=final_answers,
        contexts_formatted=final_contexts,
        metrics_df=all_results_df,
    )

    eval_logger.info("=" * 60)
    eval_logger.info("评估流程全部完成!")
    eval_logger.info(f"结果文件: {RESULT_CSV_PATH}")
    eval_logger.info("=" * 60)


def save_basic_results(
    questions: List[str],
    answers: List[str],
    contexts_formatted: List[str],
) -> None:
    """
    在 RAGAS 评估无法执行时，保存仅含 question/answer/context 的基础结果

    Args:
        questions:          问题列表
        answers:            答案列表
        contexts_formatted: 上下文列表
    """
    basic_df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "context": contexts_formatted,
    })
    basic_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
    eval_logger.info(f"基础结果（不含评估指标）已保存至: {RESULT_CSV_PATH}")


# ==============================================================================
# 测试入口
# ==============================================================================

if __name__ == "__main__":
    """
    运行方式:
        cd <项目根目录>
        uv run python eval/eval.py

    前置条件:
        1. pip install ragas datasets (或 uv pip install ragas datasets)
        2. .env 文件已配置 LLM 和嵌入模型
        3. 相关服务（Milvus, Neo4j, MongoDB 等）已启动（如需完整管线）
        4. eval/qa.csv 文件已准备好
    """
    main()
