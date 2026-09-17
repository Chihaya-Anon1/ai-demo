from __future__ import annotations

import argparse
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import BASE_DIR, get_settings

PDF_PATH = BASE_DIR / "test.pdf"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "current_affairs"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QUESTION = "习近平在上合组织提出了哪4点主张？"


def load_pdf_text(pdf_path=PDF_PATH) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise SystemExit(f"未能从 {pdf_path} 提取到文本，请确认 PDF 不是扫描件。")
    return text


def split_text(text: str) -> list[Document]:
    articles = [
        part.strip()
        for part in re.split(r"(?=\n\d+\.\S)", "\n" + text)
        if part.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    documents: list[Document] = []
    for article_id, article in enumerate(articles):
        for chunk_id, chunk in enumerate(splitter.split_text(article)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(PDF_PATH.name),
                        "article_id": article_id,
                        "chunk_id": chunk_id,
                    },
                )
            )
    return documents


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(rebuild: bool) -> Chroma:
    embeddings = build_embeddings()
    if rebuild and PERSIST_DIR.exists():
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        vectorstore.delete_collection()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    existing = vectorstore.get()
    if not existing.get("ids"):
        chunks = split_text(load_pdf_text())
        vectorstore.add_documents(chunks)
        print(f"已写入 {len(chunks)} 个文本块到 {PERSIST_DIR}")
    else:
        print(f"已加载本地向量库：{PERSIST_DIR}（{len(existing['ids'])} 条）")
    return vectorstore


def neighbor_chunks(vectorstore: Chroma, retrieved: list[Document]) -> list[str]:
    stored = vectorstore.get(include=["documents", "metadatas"])
    lookup = {
        (meta["article_id"], meta["chunk_id"]): content
        for content, meta in zip(stored["documents"], stored["metadatas"])
    }
    extras: list[str] = []
    seen = {doc.page_content for doc in retrieved}
    for doc in retrieved:
        article_id = doc.metadata.get("article_id")
        chunk_id = doc.metadata.get("chunk_id", 0)
        for nxt_id in range(chunk_id + 1, chunk_id + 6):
            nxt = lookup.get((article_id, nxt_id))
            if not nxt:
                break
            if nxt not in seen:
                extras.append(nxt)
                seen.add(nxt)
    return extras


def retrieve_context(question: str, vectorstore: Chroma) -> tuple[list[Document], str]:
    retrieved = vectorstore.similarity_search(question, k=3)
    context_parts = [doc.page_content for doc in retrieved] + neighbor_chunks(vectorstore, retrieved)
    return retrieved, "\n\n".join(context_parts)


def generate_answer(question: str, vectorstore: Chroma, *, verbose: bool = False) -> str:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先填写 .env")

    retrieved, context = retrieve_context(question, vectorstore)
    if verbose:
        print("\n===== 检索到的 3 个相关段落 =====")
        for i, doc in enumerate(retrieved, start=1):
            print(f"\n--- 段落 {i} ---\n{doc.page_content}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是时政问答助手。只根据提供的资料回答问题，不要编造资料中没有的内容。"
                "如果资料不足，请直接说明。",
            ),
            ("human", "资料：\n{context}\n\n问题：{question}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )
    result = (prompt | llm).invoke({"context": context, "question": question})
    answer = (result.content or "").strip()
    if verbose:
        print("\n===== DeepSeek 回答 =====\n")
        print(answer)
    return answer


def answer_question(question: str, vectorstore: Chroma) -> str:
    return generate_answer(question, vectorstore, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 Chroma + DeepSeek RAG Demo")
    parser.add_argument("question", nargs="?", default=None, help="要提问的问题")
    parser.add_argument("--rebuild", action="store_true", help="删除并重建本地向量库")
    args = parser.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"找不到 {PDF_PATH}，请把 PDF 放到项目根目录。")

    question = args.question or input(f"请输入问题（回车使用默认：{DEFAULT_QUESTION}）\n> ").strip()
    if not question:
        question = DEFAULT_QUESTION

    vectorstore = get_vectorstore(rebuild=args.rebuild)
    answer_question(question, vectorstore)


if __name__ == "__main__":
    main()
