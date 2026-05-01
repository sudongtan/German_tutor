from unittest.mock import MagicMock

import pytest

from src.german_tutor.tutor import generate_response


def make_llm(content):
    llm = MagicMock()
    llm.invoke.return_value.content = content
    return llm


def test_returns_content(caplog):
    llm = make_llm("Wie heißt du?")
    assert (
        generate_response([{"role": "user", "content": "hi"}], llm=llm)
        == "Wie heißt du?"
    )


def test_empty_response_returns_fallback(caplog):
    llm = make_llm("")
    result = generate_response([{"role": "user", "content": "hi"}], llm=llm)
    assert result == "I'm sorry, I couldn't generate a response. Please try again."


def test_whitespace_response_returns_fallback():
    llm = make_llm("   ")
    result = generate_response([{"role": "user", "content": "hi"}], llm=llm)
    assert result == "I'm sorry, I couldn't generate a response. Please try again."


def test_empty_response_logs_warning(caplog):
    import logging

    llm = make_llm("")
    with caplog.at_level(logging.WARNING, logger="src.german_tutor.tutor"):
        generate_response([{"role": "user", "content": "hi"}], llm=llm)
    assert "empty response" in caplog.text


def test_api_error_raises_runtime_error():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("network error")
    with pytest.raises(RuntimeError, match="LLM call failed"):
        generate_response([{"role": "user", "content": "hi"}], llm=llm)


def test_api_error_preserves_original_cause():
    llm = MagicMock()
    original = Exception("timeout")
    llm.invoke.side_effect = original
    with pytest.raises(RuntimeError) as exc_info:
        generate_response([{"role": "user", "content": "hi"}], llm=llm)
    assert exc_info.value.__cause__ is original
