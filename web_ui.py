import os

import httpx
import streamlit as st

# 前端只负责展示，检索与生成全部由 FastAPI 后端完成（真正的前后端分离）
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/ask")
BASE_URL = API_URL.rsplit("/", 1)[0]
HEALTH_URL = f"{BASE_URL}/health"

# 页面基础配置
st.set_page_config(page_title="企业制度智能问答", page_icon="📄", layout="centered")

# 样式美化
st.markdown(
    """
    <style>
    .stApp { background: #f6f7fb; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; max-width: 760px; }
    h1 { font-weight: 650; letter-spacing: 0.02em; }
    .hint { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("企业制度智能问答")
st.markdown(
    '<p class="hint">前端（Streamlit）通过 HTTP 调用后端（FastAPI）的 /ask 接口，'
    '由后端完成检索增强生成，并返回命中的原文片段用于溯源。</p>',
    unsafe_allow_html=True,
)


def ask_backend(question: str) -> tuple[str, list[str]]:
    response = httpx.post(API_URL, json={"question": question}, timeout=180.0)
    response.raise_for_status()
    data = response.json()
    return data["answer"], data.get("sources", [])


with st.sidebar:
    st.caption("后端接口")
    st.code(API_URL, language="text")
    if st.button("检查后端连接"):
        try:
            resp = httpx.get(HEALTH_URL, timeout=5.0)
            st.success(f"后端在线：{resp.json()}")
        except Exception as exc:  # noqa: BLE001 - 前端友好提示
            st.error(f"后端不可达：{exc}")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示聊天记录
chat_box = st.container()
with chat_box:
    if not st.session_state.messages:
        st.info("还没有对话。在下方输入问题后点击「提交」。")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 提问表单
with st.form("ask_form", clear_on_submit=True):
    question = st.text_input(
        "问题",
        placeholder="例如：出差住宿费的标准是多少？",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("提交", type="primary", use_container_width=True)

# 处理提交的逻辑
if submitted:
    question = (question or "").strip()
    if not question:
        st.warning("请先输入问题。")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("正在检索并生成回答..."):
            try:
                answer, sources = ask_backend(question)
            except httpx.ConnectError:
                answer = (
                    "**无法连接后端服务。**\n\n"
                    "请在项目目录另开一个终端启动后端：\n\n"
                    "```\nuvicorn app.main:app --reload\n```\n\n"
                    f"当前前端配置的接口地址：`{API_URL}`"
                )
                sources = []
            except Exception as exc:  # noqa: BLE001 - 前端友好提示
                answer = f"生成失败：{exc}"
                sources = []

        st.session_state.messages.append({"role": "assistant", "content": answer})
        if sources:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "**溯源片段（后端返回的命中原文）**\n\n"
                    + "\n\n---\n\n".join(f"> {s}" for s in sources),
                }
            )
        st.rerun()
