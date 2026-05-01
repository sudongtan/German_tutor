from langchain_openai import ChatOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(min_length=1)
    llm_model: str = "gpt-4o-mini"


settings = Settings()

LLM = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
)
