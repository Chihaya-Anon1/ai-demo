from fastapi import FastAPI, HTTPException
from httpx import HTTPStatusError

from app.deepseek import analyze_text
from app.schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="Text Analyze API", version="1.0.0")


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
