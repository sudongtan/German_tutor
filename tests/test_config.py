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
