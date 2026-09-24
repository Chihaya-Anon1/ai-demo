import asyncio

from fastapi import FastAPI, HTTPException
from httpx import HTTPStatusError
from starlette.concurrency import run_in_threadpool

from app.deepseek import analyze_text
from app.schemas import AnalyzeRequest, AnalyzeResponse, AskRequest, AskResponse

app = FastAPI(title="Text Analyze & RAG API", version="1.1.0")

# 向量库是重对象（要加载 Embedding 模型），进程内只初始化一次
_store = None
_store_lock = asyncio.Lock()


def _load_store():
    """在线程池中执行：加载 Embedding 模型并连接本地 Chroma 向量库。"""
    from rag_demo import get_vectorstore

    return get_vectorstore(rebuild=False)


async def get_store():
    global _store
    if _store is None:
        async with _store_lock:
            if _store is None:
                _store = await run_in_threadpool(_load_store)
    return _store


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "store_loaded": _store is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await analyze_text(payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="DeepSeek 鉴权失败（401）。请确认 .env 中 DEEPSEEK_API_KEY 有效，并重启 uvicorn。",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek API 调用失败: {exc.response.status_code}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="DeepSeek 返回内容无法解析为约定 JSON") from exc


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """RAG 问答：检索本地知识库 -> 拼接上下文 -> DeepSeek 生成 -> 返回答案与溯源片段。"""
    from rag_demo import answer_with_sources

    try:
        store = await get_store()
        answer, sources = await run_in_threadpool(
            answer_with_sources, payload.question, store
        )
        return AskResponse(answer=answer, sources=sources)
    except SystemExit as exc:
        raise HTTPException(status_code=500, detail=f"知识库准备失败：{exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"找不到知识库 PDF（rag_demo.PDF_PATH）：{exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="DeepSeek 鉴权失败（401）。请确认 .env 中 DEEPSEEK_API_KEY 有效，并重启 uvicorn。",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek API 调用失败: {exc.response.status_code}",
        ) from exc
