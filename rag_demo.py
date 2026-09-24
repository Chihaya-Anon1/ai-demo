from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 中文控制台默认用 GBK，打印论文原文里的连字（ﬁ U+FB01、ﬂ U+FB02）等字符会直接
# UnicodeEncodeError 崩掉整个问答流程。1 页 test.pdf 里没有这类字符，扩大到真实语料后必现。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 非常规流（被重定向/已关闭）时忽略
        pass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import BASE_DIR, get_settings

# 支持一个目录下的多份 PDF；若目录不存在则回退到单文件 test.pdf
PDF_DIR = BASE_DIR / "data"
LEGACY_PDF = BASE_DIR / "test.pdf"

PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "company_policy"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QUESTION = "请概括这份资料的主要内容。"
RETRIEVE_K = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
NEIGHBOR_WINDOW = 5
MIN_CHUNK_CHARS = 20
# ChromaDB 对单次写入有批量上限（1.5.x 实测 5461），语料一大就会报
# "Batch size of N is greater than max batch size of 5461"，必须分批写
ADD_BATCH = 1000

SYSTEM_PROMPT = (
    "你是知识库问答助手。严格根据提供的资料回答问题，不要编造资料中没有的内容。"
    "如果资料不足，请直接说明。"
)


def iter_pdf_paths() -> list[Path]:
    """收集待入库的 PDF：优先 data/ 目录，其次回退到项目根目录的 test.pdf。"""
    paths: list[Path] = []
    if PDF_DIR.exists():
        paths = sorted(p for p in PDF_DIR.glob("*.pdf"))
    if not paths and LEGACY_PDF.exists():
        paths = [LEGACY_PDF]
    if not paths:
        raise SystemExit(
            f"未找到任何 PDF。请把文档放进 {PDF_DIR}（或放置单文件 {LEGACY_PDF}）。"
        )
    return paths


def load_pdf_text(pdf_path: Path = LEGACY_PDF) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise SystemExit(f"未能从 {pdf_path} 提取到文本，请确认 PDF 不是扫描件。")
    return text


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """逐页切分，保留 source / page / chunk_id 元数据，便于检索结果溯源。"""
    reader = PdfReader(str(pdf_path))
    splitter = _splitter()
    documents: list[Document] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk_id, chunk in enumerate(splitter.split_text(text)):
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_no,
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


def _open_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def get_vectorstore(rebuild: bool = False) -> Chroma:
    """加载本地向量库；库为空（或指定 --rebuild）时按 data/ 下的 PDF 重新入库。

    注意：直接替换 PDF 不会自动重建，必须传 rebuild=True 或删除 chroma_db 目录。
    """
    embeddings = build_embeddings()
    vectorstore = _open_store(embeddings)
    existing = vectorstore.get()

    if rebuild and existing.get("ids"):
        vectorstore.delete_collection()
        vectorstore = _open_store(embeddings)
        existing = {"ids": []}

    if not existing.get("ids"):
        pdf_paths = iter_pdf_paths()
        print(f"发现 {len(pdf_paths)} 份待入库文档，开始切分与向量化……")
        documents: list[Document] = []
        for pdf_path in pdf_paths:
            docs = load_pdf_documents(pdf_path)
            documents.extend(docs)
            print(f"  {pdf_path.name}: {len(docs)} 个文本块")
        # 分批写入：ChromaDB 单批有上限（1.5.x 实测 5461），一次全丢会直接报错
        for i in range(0, len(documents), ADD_BATCH):
            vectorstore.add_documents(documents[i : i + ADD_BATCH])
            print(f"  已入库 {min(i + ADD_BATCH, len(documents))}/{len(documents)} 块", flush=True)
        print(f"已写入 {len(documents)} 个文本块到 {PERSIST_DIR}")
    else:
        print(f"已加载本地向量库：{PERSIST_DIR}（{len(existing['ids'])} 条）")

    return vectorstore


def neighbor_chunks(vectorstore: Chroma, retrieved: list[Document]) -> list[str]:
    """命中块之后，把同一页紧邻的后续块也带进上下文（减少答案被截断的概率）。

    性能说明：这里只按 (source, page) 用 metadata 过滤取「命中页」，而不是
    vectorstore.get() 全量拉取。库里有 10,643 个块时，全量拉取约 530ms/次，
    过滤后约 90ms/次（5.9x），两者产出的上下文完全一致。
    实测数据见 D:\\dsh_working\\rag-measure\\report-e2e.json。
    """
    extras: list[str] = []
    seen = {doc.page_content for doc in retrieved}
    page_cache: dict[tuple, dict[int, str]] = {}
    for doc in retrieved:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", 0)
        key = (source, page)
        if key not in page_cache:
            got = vectorstore.get(
                where={"$and": [{"source": source}, {"page": page}]},
                include=["documents", "metadatas"],
            )
            page_cache[key] = {
                meta.get("chunk_id", 0): content
                for content, meta in zip(got["documents"], got["metadatas"])
            }
        by_id = page_cache[key]
        for nxt_id in range(chunk_id + 1, chunk_id + 1 + NEIGHBOR_WINDOW):
            nxt = by_id.get(nxt_id)
            if not nxt:
                break
            if nxt not in seen:
                extras.append(nxt)
                seen.add(nxt)
    return extras


def retrieve_context(question: str, vectorstore: Chroma) -> tuple[list[Document], str]:
    retrieved = vectorstore.similarity_search(question, k=RETRIEVE_K)
    context_parts = [doc.page_content for doc in retrieved] + neighbor_chunks(vectorstore, retrieved)
    return retrieved, "\n\n".join(context_parts)


def build_llm() -> ChatOpenAI:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先填写 .env")
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "资料：\n{context}\n\n问题：{question}"),
        ]
    )


def generate_from_context(question: str, context: str) -> str:
    """只依赖 context 的生成步骤，便于被 RAG 与后端分别复用。"""
    result = (build_prompt() | build_llm()).invoke({"context": context, "question": question})
    return (result.content or "").strip()


def generate_answer(question: str, vectorstore: Chroma, *, verbose: bool = False) -> str:
    retrieved, context = retrieve_context(question, vectorstore)
    if verbose:
        print(f"\n===== 检索到的 {len(retrieved)} 个相关段落 =====")
        for i, doc in enumerate(retrieved, start=1):
            meta = doc.metadata
            print(f"\n--- 段落 {i}（{meta.get('source')} 第 {meta.get('page')} 页）---\n{doc.page_content}")

    answer = generate_from_context(question, context)
    if verbose:
        print("\n===== DeepSeek 回答 =====\n")
        print(answer)
    return answer


def answer_with_sources(question: str, vectorstore: Chroma) -> tuple[str, list[str]]:
    """供 FastAPI 后端调用：返回答案 + 命中的原文片段（带来源与页码）。"""
    retrieved, context = retrieve_context(question, vectorstore)
    answer = generate_from_context(question, context)
    sources = [
        f"[{doc.metadata.get('source')} 第 {doc.metadata.get('page')} 页] {doc.page_content}"
        for doc in retrieved
    ]
    return answer, sources


def answer_question(question: str, vectorstore: Chroma) -> str:
    return generate_answer(question, vectorstore, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 Chroma + DeepSeek RAG Demo")
    parser.add_argument("question", nargs="?", default=None, help="要提问的问题")
    parser.add_argument("--rebuild", action="store_true", help="删除并重建本地向量库")
    parser.add_argument("--stats", action="store_true", help="只打印知识库规模统计，不提问")
    args = parser.parse_args()

    vectorstore = get_vectorstore(rebuild=args.rebuild)

    if args.stats:
        stored = vectorstore.get(include=["metadatas"])
        sources = {}
        for meta in stored["metadatas"]:
            sources[meta.get("source")] = sources.get(meta.get("source"), 0) + 1
        print(f"\n集合：{COLLECTION_NAME}　文本块总数：{len(stored['ids'])}　文档数：{len(sources)}")
        for name, count in sorted(sources.items()):
            print(f"  {name:<42} {count:>5} 块")
        return

    question = args.question or input(f"请输入问题（回车使用默认：{DEFAULT_QUESTION}）\n> ").strip()
    if not question:
        question = DEFAULT_QUESTION

    answer_question(question, vectorstore)


if __name__ == "__main__":
    main()
