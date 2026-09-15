from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import BASE_DIR, get_settings


class TextSummary(BaseModel):
    summary: str = Field(description="对原文的简要总结")
    keywords: list[str] = Field(description="关键词列表")


def main() -> None:
    settings = get_settings()
    text = (BASE_DIR / "test.txt").read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("test.txt 是空的，请先写入要总结的内容。")

    parser = PydanticOutputParser(pydantic_object=TextSummary)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是文本总结助手。只输出 JSON，必须包含字段 "
                "summary（字符串）和 keywords（字符串数组）。",
            ),
            ("human", "下面是完整原文，请总结：\n\n{text}"),
        ]
    )
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    ).bind(response_format={"type": "json_object"})

    chain = prompt | llm | parser
    result = chain.invoke({"text": text})
    print(result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
