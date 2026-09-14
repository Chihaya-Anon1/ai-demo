# AI 智能文本分析与结构化提取 API

基于 FastAPI + DeepSeek 的结构化输出接口，用于提取文本摘要与关键词。

## 🛠️ 技术栈

* Python 3.10+
* FastAPI / Uvicorn
* DeepSeek API
* Pydantic (用于严格的数据校验)
* Python-dotenv

## 🚀 快速开始

**1. 克隆仓库**
git clone https://github.com/Chihaya-Anon1/ai-demo.git
cd ai-demo

**2. 创建并激活虚拟环境**
python -m venv .venv
Windows 用户执行：.\venv\Scripts\activate
Mac/Linux 用户执行：source .venv/bin/activate

**3. 安装依赖**
pip install -r requirements.txt

**4. 配置环境变量**
复制 .env.example 为 .env，并填入你的 DEEPSEEK_API_KEY。

**5. 启动服务**
uvicorn app.main:app --reload

## 📝 接口说明

* POST /analyze：接收一段文本，返回结构化的摘要与关键词。
