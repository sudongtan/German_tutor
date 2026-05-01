"""
LangGraph graph for topic-based German Q&A sessions.

First invocation: invoke({"user_topic": str, "user_level": str}, config).
Subsequent turns: invoke(Command(resume=user_answer), config).
Exposes `app` (compiled graph) and `State` for external use.
"""

import logging
import operator
import sqlite3
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.german_tutor.config import LLM, settings
from src.german_tutor.prompts import (
    AVOID_REPEAT,
    GENERATE_EXAMPLES,
    GENERATE_EXPRESSIONS,
    GENERATE_FEEDBACK,
    GENERATE_QUESTIONS_BATCH,
)
from src.german_tutor.schemas import FeedbackResult, QuestionBatch

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


class State(BaseModel):
    user_topic: str = ""
    user_level: str = ""
    ai_question: str = ""
    user_answer: str = ""
    feedback_corrected_answer: str = ""
    feedback_explanation: str = ""
    feedback_score: int = 0
    hint: str = ""
    moved_to_next: bool = False
    question_pool: list[str] = Field(default_factory=list)
    asked_questions: Annotated[list[str], operator.add] = Field(default_factory=list)
    answers: Annotated[list[str], operator.add] = Field(default_factory=list)
    feedbacks: Annotated[list[dict], operator.add] = Field(default_factory=list)


def generate_question(state: State):
    pool = list(state.question_pool)

    if not pool:
        avoid = (
            AVOID_REPEAT.format(previous=state.asked_questions)
            if state.asked_questions
            else ""
        )
        content = GENERATE_QUESTIONS_BATCH.format(
            topic=state.user_topic, level=state.user_level, avoid=avoid
        )
        result = generate_structured_response(
            [{"role": "system", "content": content}], QuestionBatch
        )
        pool = result.questions

    question = pool.pop(0)
    return {
        "ai_question": question,
        "asked_questions": [question],
        "question_pool": pool,
        "hint": "",
        "moved_to_next": True,
    }


def ask_for_answer(state: State):
    user_answer = interrupt(state.ai_question)
    return {"user_answer": user_answer}


def route_input(state: State):
    answer = state.user_answer.strip().lower()
    if answer == "stop":
        return "stop"
    if answer == "next":
        return "next"
    if answer == "1":
        return "examples"
    if answer == "2":
        return "expressions"
    return "answer"


def generate_examples(state: State):
    examples = generate_response(
        [
            {
                "role": "system",
                "content": GENERATE_EXAMPLES.format(
                    question=state.ai_question, level=state.user_level
                ),
            }
        ]
    )
    return {"hint": examples}


def generate_expressions(state: State):
    expressions = generate_response(
        [
            {
                "role": "system",
                "content": GENERATE_EXPRESSIONS.format(
                    question=state.ai_question, level=state.user_level
                ),
            }
        ]
    )
    return {"hint": expressions}


def generate_feedback(state: State):
    result = generate_structured_response(
        [
            {
                "role": "system",
                "content": GENERATE_FEEDBACK.format(level=state.user_level),
            },
            {"role": "user", "content": state.user_answer},
        ],
        FeedbackResult,
    )
    return {
        "feedback_corrected_answer": result.corrected_answer,
        "feedback_explanation": result.explanation,
        "feedback_score": result.score,
        "answers": [state.user_answer],
        "feedbacks": [result.model_dump()],
        "moved_to_next": False,
    }


graph = StateGraph(State)
graph.add_node("generate_question", generate_question)
graph.add_node("ask_for_answer", ask_for_answer)
graph.add_node("generate_examples", generate_examples)
graph.add_node("generate_expressions", generate_expressions)
graph.add_node("generate_feedback", generate_feedback)

graph.set_entry_point("generate_question")
graph.add_edge("generate_question", "ask_for_answer")
graph.add_conditional_edges(
    "ask_for_answer",
    route_input,
    {
        "stop": END,
        "next": "generate_question",
        "examples": "generate_examples",
        "expressions": "generate_expressions",
        "answer": "generate_feedback",
    },
)
graph.add_edge("generate_examples", "ask_for_answer")
graph.add_edge("generate_expressions", "ask_for_answer")
graph.add_edge("generate_feedback", "ask_for_answer")

_conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(_conn))
