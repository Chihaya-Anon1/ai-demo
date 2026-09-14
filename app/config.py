from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    @field_validator("deepseek_api_key", "deepseek_base_url", "deepseek_model", mode="before")
    @classmethod
    def strip_values(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().strip('"').strip("'")
        return value


def get_settings() -> Settings:
    load_dotenv(ENV_FILE, override=True)
    return Settings()
