# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

基石智库 — a Chinese-language RAG system for product manuals (PDF/Markdown). It has two independent pipelines, each built as a **LangGraph** state graph fronted by its own **FastAPI** server:

- **import_process** — ingest a document → parse → split → embed → store in Milvus + Neo4j knowledge graph.
- **query_process** — answer a user question via multi-route retrieval → fusion → rerank → LLM generation.

Code is heavily commented in Chinese; match that style when editing.

## Commands

This project uses **uv** (see `pyproject.toml`, `uv.lock`). Python 3.12 (`.python-version`), `requires-python >=3.11`.

```powershell
uv sync                      # install dependencies into .venv

# Run the two services (separate processes/terminals):
uv run python -m app.import_process.api.server   # import service  -> http://127.0.0.1:8000  (docs at /docs, UI at /import/html)
uv run python -m app.query_process.api.server    # query service   -> http://0.0.0.0:8001    (UI at /query/html)

# Download local models (run once; targets paths under D:/ai_models, override in the scripts):
uv run python app/tool/download_bgem3.py         # BGE-M3 embedding model
uv run python app/tool/download_reranker.py      # BGE reranker model
```

**Tests:** there is no test framework (no pytest). Each module is exercised via its own `if __name__ == "__main__":` block — run a file directly to test it, e.g. `uv run python -m app.import_process.agent.main_graph` runs the full import pipeline against a sample PDF in `doc/`. `main.py` is an empty placeholder; do not use it as an entry point.

## Configuration

All runtime config comes from a `.env` file at the project root, loaded via `python-dotenv` (`load_dotenv()` is called at the top of every `app/conf/*.py` and several other modules). `.env` is git-ignored. **It currently contains real API keys and remote service credentials (DashScope/Bailian, Milvus, MongoDB, MinIO, MinerU, Neo4j) — treat it as a secret and never commit or echo it.**

`app/conf/` holds one dataclass per external system, each reading env vars at import time: `lm_config` (DashScope/Qwen LLM + VL model, OpenAI-compatible), `embedding_config` (BGE-M3), `reranker_config`, `milvus_config`, `minio_config`, `mineru_config`, `bailian_mcp_config` (DashScope WebSearch MCP), `neo4j_conf` (Neo4j graph DB). Add new external config the same way.

Notable env vars: `LLM_DEFAULT_MODEL` (qwen-flash), `VL_MODEL` (qwen3-vl-flash), `OPENAI_BASE_URL`/`OPENAI_API_KEY` (DashScope compatible endpoint), `BGE_M3_PATH` + `BGE_DEVICE`/`BGE_FP16` (local model; CPU requires FP16=0), `MILVUS_URL`, `CHUNKS_COLLECTION` (`rag_chunks`), `ITEM_NAME_COLLECTION` (`rag_item_names`), `MONGO_URL`/`MONGO_DB_NAME`, `MINIO_*`, `MCP_DASHSCOPE_BASE_URL`, `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`/`NEO4J_DATABASE`, `ENTITY_NAME_COLLECTION`.

## Architecture

### LangGraph pipelines

Both pipelines follow the same conventions, which are the key thing to understand before editing:

- A typed `state.py` (`TypedDict`) defines the shared dict that flows through every node, plus `create_*_default_state(**overrides)` / `get_*_default_state()` helpers (deep-copy a default template). Each node receives the state, mutates/returns fields, and the graph merges them.
- `agent/nodes/node_*.py` — one file per node, each a function `node_xxx(state) -> dict`.
- `agent/main_graph.py` — wires nodes into a `StateGraph`, sets the entry point, adds static and conditional edges, and `.compile()`s to a module-level app object (`kb_import_app` / `query_app`). The API server imports that compiled app and calls `.invoke(state)`.

**Import graph** (`app/import_process/agent/main_graph.py`, compiled as `kb_import_app`):
`node_entry` → (conditional on file type: `is_md_read_enabled` → `node_md_img`, `is_pdf_read_enabled` → `node_pdf_to_md`, else END) → `node_pdf_to_md` → `node_md_img` → `node_document_split` → `node_item_name_recognition` → `node_bge_embedding` → `node_import_milvus` → `node_import_neo4j` → END. PDF→Markdown uses the MinerU API; images extracted from Markdown are summarized by the VL model and uploaded to MinIO. The final step extracts entities and relations from chunks via LLM and writes them to Neo4j as a knowledge graph.

Import nodes may optionally extend `app/import_process/base.py:BaseNode`, which provides a shared logger, Neo4j config, and an ABC `process` method. Nodes using `BaseNode` decompose their logic into `step_1_xxx`, `step_2_xxx`, etc. decorated with `@step_log("step_name")`.

**Query graph** (`app/query_process/agent/main_graph.py`, compiled as `query_app`):
`node_item_name_confirm` is the entry. Its conditional router (`node_item_name_confirm_after_router`) checks `state['answer']`: if non-empty (empty question / ambiguous or unrecognized product), it short-circuits straight to `node_answer_output`; otherwise it **fans out to three retrieval nodes in parallel** — `node_search_embedding` (dense+sparse vector search), `node_search_embedding_hyde` (HyDE hypothetical-doc retrieval), `node_web_search_mcp` (Bailian WebSearch via MCP). All three converge on `node_rrf` (Reciprocal Rank Fusion of the dual vector routes) → `node_rerank` (BGE reranker scoring with a "断崖"/cliff cutoff algorithm to drop low-relevance chunks) → `node_answer_output` (assembles prompt, calls LLM, streams answer, extracts inline images from chunks, saves chat history to MongoDB) → END.

### Cross-cutting layers

- `app/clients/` — singleton-style clients for external stores: `milvus_utils` (`get_milvus_client()`, hybrid `AnnSearchRequest`/`WeightedRanker` search), `minio_utils`, `mongo_history_utils` (chat history CRUD + **memory compression pipeline**), `neo4j_utils` (Neo4j driver, connectivity-verified at init).
- `app/lm/` — model wrappers: `lm_utils.get_llm_client(model, json_mode)` returns a cached `ChatOpenAI` (DashScope-compatible; injects `enable_thinking=False` for Qwen, optional JSON mode), `embedding_utils` (BGE-M3 dense+sparse), `reranker_utils`.
- `app/core/` — `logger` (loguru, configured from env, with `@node_log(name)` / `@step_log(name)` decorators used on graph nodes; also re-exports `PROJECT_ROOT`) and `load_prompt(name, **kwargs)` which loads & `.format()`-renders `prompts/<name>.prompt`.
- `prompts/` — externalized prompt templates (`.prompt` files with `{placeholder}` vars). Add new prompts here and load via `load_prompt`. Current prompts: `answer_out`, `image_summary`, `hyde_prompt`, `item_name_recognition`, `rewritten_query_and_itemnames`, `product_recognition_system`, `memory_compress_agent`.
- `app/utils/` — `path_util.PROJECT_ROOT` (root discovery via `.env` marker), `task_utils` (in-memory per-`task_id`/`session_id` status tracking + node-name→Chinese display map; **single-process only**), `sse_utils` (SSE queue + `sse_generator` for streaming), plus Milvus string escaping / sparse-vector normalization helpers.

### Request flow (query service)

`POST /query` with `{query, session_id, is_stream}`. Streaming mode creates an SSE queue, runs `run_query_graph` as a FastAPI `BackgroundTask`, and the client consumes `GET /stream/{session_id}`; a terminal `FINAL` SSE event carries the answer + image URLs and closes the stream. Non-streaming runs the graph inline and returns the answer in the response. History endpoints (`GET/DELETE /history/{session_id}`) are backed by MongoDB. The import service mirrors this: `POST /upload` saves files under `output/YYYYMMDD/<task_id>/`, runs the import graph in the background, and the client polls `GET /status/{task_id}`.

### MongoDB memory compression

`mongo_history_utils.py` includes a **conversation memory compression pipeline** (`trigger_memory_compress_pipeline`). When a session accumulates more chat messages than a threshold (default 15), it invokes an LLM via the `memory_compress_agent.prompt` template to produce:
- A **structured summary** of compressed conversation history
- A list of **user constraints/preferences** (hard rules extracted from chat)

These are stored in the `chat_summary` MongoDB collection (one document per `session_id`, upserted). Old raw messages are then garbage-collected, keeping only the most recent tail (default 6). The compressed memory can be injected back into future LLM contexts via `get_llm_context_messages()`.

This should be called as a `BackgroundTask` after the answer output node completes (not currently wired into the graph — it's opt-in per-invocation).

## Conventions

- **Adding a graph node:** create `agent/nodes/node_<x>.py` with a `node_<x>(state)` function, register it in `main_graph.py` with `add_node`/edges, and add any new state fields to `state.py` (and its default template). If the node is user-visible, add its Chinese display name to `_NODE_NAME_TO_CN` in `app/utils/task_utils.py`.
  - For import-process nodes that need Neo4j config, subclass `app.import_process.base.BaseNode` and decompose logic into `step_1_xxx`, `step_2_xxx` methods.
- Use the shared `logger` from `app.core.logger` everywhere; decorate nodes with `@node_log("node_name")` and internal steps with `@step_log("step_name")`.
- External clients are module-level singletons guarded by a global — reuse the getter rather than constructing new connections.
- Neo4j operations use Cypher queries via `neo4j_utils.get_neo4j_driver()`. The driver verifies connectivity at init time (fail-fast on bad credentials).
- Imports are absolute from the `app.` package root; run servers as modules (`python -m app...`) so these resolve.
- Prompts are externalized in `prompts/*.prompt` as format-string templates. Load them with `load_prompt(name, **kwargs)` — never hardcode prompt text in Python.
