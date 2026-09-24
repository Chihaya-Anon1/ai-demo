# 企业智能知识库 RAG 问答系统

基于 LangChain + ChromaDB + DeepSeek 的本地私有知识库问答系统。支持 PDF 解析、向量化存储、语义检索与大模型生成回答。

## 🏗️ 系统架构

用户提问 (Streamlit UI) 
  -> LangChain 检索器 
  -> ChromaDB 向量库 (返回最相关 3 个文本块) 
  -> 拼接 Prompt 
  -> DeepSeek API 生成回答 
  -> Pydantic 结构化校验 
  -> 返回前端

## 🛠️ 技术栈
- 后端服务：Python / FastAPI
- AI框架：LangChain / Pydantic
- 大模型：DeepSeek API (Chat)
- 向量化：HuggingFace Embeddings (本地模型)
- 向量数据库：ChromaDB
- 前端界面：Streamlit

## 📁 项目结构
- app/ - FastAPI 后端服务代码
- rag_demo.py - RAG 核心逻辑（文档处理、向量检索、LLM调用）
- web_ui.py - Streamlit 交互式前端
- requirements.txt - 项目依赖

## 🚀 快速开始
1. 安装依赖：pip install -r requirements.txt
2. 配置环境变量：复制 .env.example 为 .env，填入你的 DEEPSEEK_API_KEY
3. 启动后端：uvicorn app.main:app --reload
4. 启动前端：streamlit run web_ui.py
5. 浏览器访问 http://localhost:8501 即可开始问答。

## 📊 性能实测（本地 CPU，34 份公开技术论文语料）

| 指标 | 实测值 |
|---|---|
| 语料规模 | 34 份论文 / **949 页** / 322.7 万字 → **10,643 个文本块** |
| 全量向量化 + 入库 | **1277 秒（21.3 分钟）**，8.3 块/秒（CPU 10 线程） |
| 检索耗时 | Top-3 平均 **65~75ms**（P95 91ms） |
| 检索命中率（文档级） | Top-1 **94.1%** / Top-3 **97.1%**（33/34） |
| 端到端（检索 + DeepSeek 生成） | 平均 **0.97 秒**（P50 0.82s / P95 2.15s） |
| 冷启动模型加载 | **47 秒**（本地缓存 + 离线模式） |

> 语料为公开论文（Transformer / BERT / GPT-3 / GPT-4 / Llama2 / CLIP / LoRA / RAG / DDPM / Whisper 等），仅用于本地评测，**不随仓库分发**（见 `.gitignore` 的 `data/`）。

## 🚧 踩坑与排错记录（工程实战经验）

以下都是开发中真实遇到并解决的问题，而非"只调了个 API"：

1. **语料规模化后 ChromaDB 写入直接崩**
   问题：一次性 `add_documents(10643 个块)` 报 `InternalError: Batch size of 10643 is greater than max batch size of 5461`。
   原因：ChromaDB 单次写入有硬上限，1 页 PDF（2 个块）时永远暴露不出来。
   解决：分片写入（`ADD_BATCH = 1000`）并打印进度，34 份文档全量入库跑通。

2. **检索链路性能问题：每次查询全量拉取全库**
   问题：`neighbor_chunks()` 为取"同页后续块"，每次提问都执行 `vectorstore.get()` 把**整个知识库**拉进内存重建索引。
   实测：10,643 块规模下 **532ms/次**（占端到端耗时一半以上）。
   解决：改为按 `source + page` 做 metadata 过滤，只取命中页并做页内缓存 → **90ms/次，5.9 倍提速**，且逐条比对证明**产出的上下文与原来完全一致**（零功能回归）。

3. **HuggingFace 模型加载假死**
   问题：国内网络下 `sentence-transformers` 联网校验模型会卡住——进程活着但 CPU 不动，十几分钟零产出。
   解决：改用本地缓存 + 离线模式（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），冷启动稳定在 **47s**，不再受网络波动影响。

4. **Streamlit 页面刷新导致模型重复加载**
   问题：每次交互刷新都重新初始化向量模型，页面卡死。
   解决：用 `@st.cache_resource` 缓存向量库单例。

5. **Python 依赖地狱**
   问题：安装 `langchain-chroma` 等库时与原有 streamlit / tensorflow 依赖冲突。
   解决：`pip install --no-deps` 绕过依赖解析安装，再手动补齐缺失组件。