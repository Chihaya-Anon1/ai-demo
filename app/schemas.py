from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="待分析的原始文本")


class AnalyzeResponse(BaseModel):
    summary: str = Field(description="文本摘要")
    keywords: list[str] = Field(description="关键词列表")
