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

## 🚧 踩坑与排错记录（工程实战经验）
在开发过程中，我作为开发者解决了以下实际工程问题，而非仅仅“调用API”：

1. HuggingFace 模型下载超时
   问题：由于本地网络限制，sentence-transformers 模型下载频繁报 WinError 10060。
   解决：通过配置环境变量 HF_ENDPOINT="https://hf-mirror.com" 接入国内镜像源，实现秒级加载模型。

2. Streamlit 页面刷新导致模型重复加载
   问题：每次用户交互刷新页面时，系统都会重新初始化向量模型，导致页面卡死。
   解决：使用 Streamlit 的 @st.cache_resource 装饰器缓存向量库单例，彻底解决重复加载导致的阻塞问题。

3. Python 依赖地狱
   问题：全局环境安装 langchain-chroma 等库时，与原有的 streamlit、tensorflow 发生严重依赖冲突。
   解决：利用 pip install --no-deps 绕过依赖解析强行安装，并手动补充缺失的组件（如 sentence-transformers），最终跑通全链路。