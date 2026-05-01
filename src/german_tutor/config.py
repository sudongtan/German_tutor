import os

from langchain_openai import ChatOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(min_length=1)
    llm_model: str = "gpt-4o-mini"
    whisper_model: str = "whisper-1"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "german-tutor"
    langchain_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()

if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

LLM = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
)
