# PDF

### DeepDoc (默认)
- RAGFlow 自研的视觉模型，全套执行 `OCR（文字识别）`、`TSR（表格结构识别）`和 `DLR（文档布局分析）`任务

### Naive
- PDF 全是**纯文本文件**，选此项跳过所有 OCR 和布局分析，极速提取


### MinerU
- 集成自开源工具
- RAGFlow 作为远程客户端，配置环境变量（如 `MINERU_BACKEND` 设为 `vlm-vllm-engine` 等）
- 异步调用外部的 `MinerU FastAPI` 服务，利用大视觉模型（VLM）进行高质量重构

### Docling
- 集成 IBM 开源的 Docling
- 支持本地进程解析，也支持配置 `DOCLING_SERVER_URL`
- 调用外部的 `Docling Serve` 实例，最终转换为 `markdown/text`

### 第三方视觉模型
- 调用外部特定模型供应商的视觉大模型


# 电子表格（Spreadsheet：XLSX, XLS, CSV）
- **输出格式**：`html`
- **解析策略**：专门**保护表格的原始物理布局**和**单元格结构**，将其整体**转化为 HTML 表格标签**
- 避免表格数据在纯文本化时错位


# 图片（Image：PNG, JPG, JPEG, GIF, TIF）
- 解析策略：默认使用系统自带的原生 `OCR 模型`进行文字提取
- 同时允许用户在配置好模型供应商后，切换为更强大的 `VLM（视觉大模型）`读图


# 纯文本与标记语言（Text & Markup：TXT, MD, MDX, HTML, JSON）
- 输出格式：`text`
- 解析策略：执行去标签化（Stripping）。自动剥离 HTML 的标签、Markdown 的语法符号等，只留下干净无格式纯文本

# Word（DOCX）
- 输出格式：`json`
- 解析策略：结构化提取
- 完整保留原文档的层级树状信息，包括`标题 Titles`、`段落 Paragraphs`、`表格 Tables`以及`页眉页脚 Headers/Footers`


# PowerPoint 演示文稿（PPTX, PPT）
- 输出格式：`json`
- 解析策略：按`幻灯片 Slide`逐页切分。在 JSON 中会严格区分并标记每一页的“`标题`、`正文文本`以及`备注 Notes`