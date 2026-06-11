# 掌柜智库 — RAG 智能知识库系统

基于 **LangGraph** + **FastAPI** 构建的企业级 RAG（检索增强生成）系统，面向产品手册（PDF/Markdown）的智能导入与问答。支持多路混合检索、知识图谱增强、流式对话和完整的 RAGAS 评估体系。

## 架构总览

```
┌─────────────────────────────────────────────────────┐
│                   Vue 3 前端 (Vite)                  │
│         /import (文档上传)  /query (智能问答)         │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
     ┌─────────▼──────┐  ┌───────▼──────────┐
     │ 导入服务 :8000 │  │ 查询服务 :8001   │
     │ FastAPI        │  │ FastAPI          │
     │ LangGraph      │  │ LangGraph        │
     └────────┬───────┘  └───────┬──────────┘
              │                  │
    ┌─────────▼──────────────────▼───────────┐
    │           外部存储 & 模型层              │
    │  Milvus │ Neo4j │ MongoDB │ MinIO     │
    │  BGE-M3 │ BGE Reranker │ Qwen LLM    │
    └────────────────────────────────────────┘
```

系统包含两条独立的 **LangGraph** 管线，各自前端一个 **FastAPI** 服务：

### 导入管线 (Import Pipeline) → `:8000`

```
上传文件 → 文件校验 → PDF→MD (MinerU) → 图片摘要 (VL) → 文档分块
→ 产品名识别 → BGE-M3 向量化 → 写入 Milvus → 实体/关系抽取 → 写入 Neo4j
```

### 查询管线 (Query Pipeline) → `:8001`

```
问题输入 → 产品名消歧 → ┬─ 稠密+稀疏向量检索
                        ├─ HyDE 假设文档检索
                        ├─ KG 知识图谱检索       → RRF 融合 → BGE 重排序
                        └─ 网络搜索 (MCP)              ↓
                                                  LLM 答案生成
                                                  (SSE 流式输出)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 框架 | LangGraph (状态图编排) + FastAPI (HTTP/SSE) |
| LLM | Qwen-Flash / Qwen3-VL-Flash (DashScope, OpenAI 兼容) |
| 嵌入模型 | BGE-M3 (本地部署, 稠密+稀疏双向量) |
| 重排序 | BGE Reranker Large (本地部署) |
| 向量库 | Milvus (混合检索: 稠密 + 稀疏) |
| 图数据库 | Neo4j (知识图谱实体/关系存储) |
| 对象存储 | MinIO (文档内嵌图片) |
| 对话存储 | MongoDB (聊天历史 + 记忆压缩) |
| PDF 解析 | MinerU API |
| 前端 | Vue 3 + Vite |
| 评估 | RAGAS (忠实度/相关性/精确度/召回率/正确性) |

## 环境要求

- **Python** ≥ 3.11（推荐 3.12）
- **uv** 包管理器
- **Milvus** 向量数据库
- **Neo4j** 图数据库
- **MongoDB** 文档数据库
- **MinIO** 对象存储
- **MinerU** PDF 解析服务

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/effson/RAG_Pro.git
cd RAG_Pro
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 下载本地模型

```bash
uv run python app/tool/download_bgem3.py      # BGE-M3 嵌入模型
uv run python app/tool/download_reranker.py    # BGE 重排序模型
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件（参考下方配置说明）。

### 5. 启动服务

```bash
# 终端 1 — 导入服务
uv run python -m app.import_process.api.server    # → http://127.0.0.1:8000

# 终端 2 — 查询服务
uv run python -m app.query_process.api.server     # → http://0.0.0.0:8001
```

### 6. 打开前端

- 导入页面: http://127.0.0.1:8000/import/html
- 问答页面: http://127.0.0.1:8001/query/html

或启动 Vite 开发服务器:

```bash
cd front && npm install && npm run dev
```

## 环境变量配置 (.env)

```bash
# ── LLM (DashScope / OpenAI 兼容) ──
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_api_key_here
LLM_DEFAULT_MODEL=qwen-flash
LLM_DEFAULT_TEMPERATURE=0.1
VL_MODEL=qwen3-vl-flash

# ── BGE-M3 嵌入模型 ──
BGE_M3_PATH=D:/ai_models/bge-m3
BGE_DEVICE=cuda:0
BGE_FP16=1

# ── BGE 重排序模型 ──
BGE_RERANKER_LARGE=D:/ai_models/bge-reranker-large
BGE_RERANKER_DEVICE=cuda:0
BGE_RERANKER_FP16=1

# ── Milvus ──
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=rag_chunks
ITEM_NAME_COLLECTION=rag_item_names

# ── Neo4j ──
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# ── MongoDB ──
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=rag_db

# ── MinIO ──
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=rag-base-files

# ── MinerU PDF 解析 ──
MINERU_API_URL=http://localhost:8888

# ── DashScope WebSearch MCP ──
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com
ENTITY_NAME_COLLECTION=rag_entity_names
```

## 项目结构

```
RAG_Pro/
├── app/
│   ├── clients/              # 外部客户端 (Milvus/MinIO/MongoDB/Neo4j)
│   ├── conf/                 # 配置数据类 (一个外部系统一个文件)
│   ├── core/                 # 核心 (logger, prompt 加载器)
│   ├── import_process/       # 导入管线
│   │   ├── agent/
│   │   │   ├── main_graph.py # StateGraph → kb_import_app
│   │   │   ├── state.py      # 导入状态 TypedDict
│   │   │   └── nodes/        # 8 个管线节点
│   │   └── api/server.py     # FastAPI :8000
│   ├── lm/                   # 模型封装 (LLM/Embedding/Reranker)
│   ├── query_process/        # 查询管线
│   │   ├── agent/
│   │   │   ├── main_graph.py # StateGraph → query_app
│   │   │   ├── state.py      # 查询状态 TypedDict
│   │   │   └── nodes/        # 8 个管线节点
│   │   └── api/server.py     # FastAPI :8001
│   ├── tool/                 # 工具脚本 (模型下载)
│   └── utils/                # 工具函数 (SSE/任务状态/路径)
├── prompts/                  # LLM 提示词模板 (.prompt)
├── eval/                     # RAGAS 评估
│   ├── eval.py               # 评估脚本
│   ├── qa.csv                # QA 数据集 (question + ground_truth)
│   └── qa_result.csv         # 评估结果
├── front/                    # Vue 3 前端
├── doc/                      # 待导入的 PDF 产品手册
├── pyproject.toml            # 依赖配置
└── uv.lock                   # 依赖锁文件
```

## 评估

项目内置了基于 **RAGAS** 框架的完整评估体系，支持 5 项指标：

| 指标 | 说明 |
|------|------|
| **Faithfulness** | 忠实度/幻觉度 — 答案是否忠于检索上下文 |
| **Answer Relevance** | 答案相关性 — 答案与问题的语义匹配度 |
| **Context Precision** | 上下文精确度 — 检索文档排序质量 |
| **Context Recall** | 上下文召回率 — 上下文对参考答案的覆盖度 |
| **Answer Correctness** | 答案正确性 — 答案的事实准确性和语义正确性 |

```bash
# 安装评估依赖
uv pip install ragas datasets openai

# 运行评估
uv run python eval/eval.py
```

输入 `eval/qa.csv`（question + ground_truth），输出 `eval/qa_result.csv`（含所有指标得分）。

## 开发约定

- **添加新节点**: 在 `agent/nodes/node_<name>.py` 中创建函数 `node_<name>(state)`，在 `main_graph.py` 中注册
- **日志**: 使用 `@node_log("name")` / `@step_log("step")` 装饰器
- **提示词**: 统一放在 `prompts/` 目录，通过 `load_prompt(name, **kwargs)` 加载
- **配置**: 每个外部系统在 `app/conf/` 中有一个对应的数据类
- **测试**: 每个模块通过 `if __name__ == "__main__":` 块独立测试，无 pytest

## 安全提醒

- **`.env` 文件包含真实 API 密钥，已被 `.gitignore` 排除，切勿提交到版本控制**
- 生产部署时请启用 HTTPS 并配置防火墙规则
- 定期更新依赖以修复安全漏洞

## License

内部项目，仅供团队使用。
