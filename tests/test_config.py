import pytest
from pydantic import ValidationError

from src.german_tutor.config import Settings


def test_missing_api_key_raises():
    with pytest.raises((ValidationError, Exception)):
        Settings(openai_api_key="")


def test_default_model():
    s = Settings(openai_api_key="test-key")
    assert s.llm_model == "gpt-4o-mini"


def test_model_can_be_overridden():
    s = Settings(openai_api_key="test-key", llm_model="gpt-4o")
    assert s.llm_model == "gpt-4o"


def test_langsmith_tracing_can_be_enabled():
    s = Settings(openai_api_key="test-key", langchain_tracing_v2=True)
    assert s.langchain_tracing_v2 is True


def test_langsmith_tracing_can_be_disabled():
    s = Settings(openai_api_key="test-key", langchain_tracing_v2=False)
    assert s.langchain_tracing_v2 is False


def test_langsmith_api_key_can_be_set():
    s = Settings(openai_api_key="test-key", langchain_api_key="ls__test")
    assert s.langchain_api_key == "ls__test"


def test_langsmith_project_can_be_set():
    s = Settings(openai_api_key="test-key", langchain_project="my-project")
    assert s.langchain_project == "my-project"
