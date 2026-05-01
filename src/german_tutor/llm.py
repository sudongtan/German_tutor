"""
Shared LLM utilities used by all graphs.

Exposes `generate_response` and `generate_structured_response`, which wrap
LangChain invocations with consistent error handling and fallback logging.
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.german_tutor.config import LLM

logger = logging.getLogger(__name__)


def generate_response(
    messages: list[dict],
    llm: BaseChatModel | None = None,
) -> str:
    model = llm or LLM
    try:
        content = model.invoke(messages).content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    if not content or not content.strip():
        logger.warning("LLM returned an empty response for messages: %s", messages)
        return "I'm sorry, I couldn't generate a response. Please try again."

    return content


def generate_structured_response(
    messages: list[dict],
    schema: type,
    llm: BaseChatModel | None = None,
) -> object:
    model = llm or LLM
    try:
        return model.with_structured_output(schema).invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e
