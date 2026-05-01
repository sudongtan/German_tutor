from unittest.mock import MagicMock

import pytest

from src.german_tutor.schemas import QuestionBatch
from src.german_tutor.tutor import generate_response, generate_structured_response


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


class TestGenerateStructuredResponse:
    def make_structured_llm(self, return_value):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = return_value
        return llm

    def test_returns_schema_instance(self):
        batch = QuestionBatch(questions=["Q1", "Q2"])
        llm = self.make_structured_llm(batch)
        result = generate_structured_response(
            [{"role": "user", "content": "test"}], QuestionBatch, llm=llm
        )
        assert result == batch

    def test_passes_schema_to_with_structured_output(self):
        batch = QuestionBatch(questions=["Q1"])
        llm = self.make_structured_llm(batch)
        generate_structured_response(
            [{"role": "user", "content": "test"}], QuestionBatch, llm=llm
        )
        llm.with_structured_output.assert_called_once_with(QuestionBatch)

    def test_raises_runtime_error_on_failure(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = Exception(
            "API error"
        )
        with pytest.raises(RuntimeError, match="LLM call failed"):
            generate_structured_response(
                [{"role": "user", "content": "test"}], QuestionBatch, llm=llm
            )

    def test_preserves_original_cause(self):
        llm = MagicMock()
        original = Exception("timeout")
        llm.with_structured_output.return_value.invoke.side_effect = original
        with pytest.raises(RuntimeError) as exc_info:
            generate_structured_response(
                [{"role": "user", "content": "test"}], QuestionBatch, llm=llm
            )
        assert exc_info.value.__cause__ is original
