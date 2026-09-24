from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, description="待分析的原始文本")


class AnalyzeResponse(BaseModel):
    summary: str = Field(description="文本摘要")
    keywords: list[str] = Field(description="关键词列表")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="针对本地知识库的问题")


class AskResponse(BaseModel):
    answer: str = Field(description="基于知识库内容生成的回答")
    sources: list[str] = Field(default_factory=list, description="命中的原文片段，用于溯源")
