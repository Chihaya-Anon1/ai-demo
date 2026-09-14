import json

import httpx

from app.config import get_settings
from app.schemas import AnalyzeResponse

SYSTEM_PROMPT = """你是文本分析助手。根据用户输入的文本，返回 JSON：
{"summary": "摘要", "keywords": ["词1", "词2"]}
要求：
- summary 为简洁中文摘要
- keywords 为 3 到 8 个关键词
- 只输出 JSON，不要其它文字"""


async def analyze_text(text: str) -> AnalyzeResponse:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填入密钥")

    url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.deepseek_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return AnalyzeResponse.model_validate(parsed)
