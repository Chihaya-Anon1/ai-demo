import streamlit as st
from rag_demo import generate_answer, get_vectorstore

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
st.markdown('<p class="hint">基于本地 PDF 知识库与 DeepSeek，对企业制度资料进行检索问答。</p>', unsafe_allow_html=True)

# 缓存向量库，避免每次刷新都重新加载模型
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    return get_vectorstore(rebuild=False)

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 加载知识库（加个按钮或者加载状态提示，首次加载可能非常慢）
try:
    with st.spinner("正在加载本地知识库（首次加载需下载模型，请耐心等待 1-3 分钟）..."):
        vectorstore = load_vectorstore()
except Exception as exc:
    st.error(f"知识库加载失败：{exc}")
    st.stop()

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
        placeholder="例如：出差住宿费的标准是多少？", # 已修正为你最新 PDF 的测试问题
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
                answer = generate_answer(question, vectorstore)
            except Exception as exc:
                answer = f"生成失败：{exc}"
                
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()