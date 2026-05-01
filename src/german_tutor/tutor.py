import logging
import operator
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.german_tutor.config import LLM
from src.german_tutor.prompts import AVOID_REPEAT, GENERATE_FEEDBACK, GENERATE_QUESTION

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


class State(BaseModel):
    user_topic: str = ""
    ai_question: str = ""
    user_answer: str = ""
    ai_feedback: str = ""
    asked_questions: Annotated[list[str], operator.add] = Field(default_factory=list)
    answers: Annotated[list[str], operator.add] = Field(default_factory=list)
    feedbacks: Annotated[list[str], operator.add] = Field(default_factory=list)


def generate_question(state: State):
    avoid = (
        AVOID_REPEAT.format(previous=state.asked_questions)
        if state.asked_questions else ""
    )
    content = GENERATE_QUESTION.format(topic=state.user_topic, avoid=avoid)
    question = generate_response([{"role": "system", "content": content}])
    return {"ai_question": question, "asked_questions": [question]}


def ask_for_answer(state: State):
    user_answer = interrupt(state.ai_question)
    return {"user_answer": user_answer}


def should_stop(state: State):
    if state.user_answer.strip().lower() == "stop":
        return "stop"
    return "continue"


def generate_feedback(state: State):
    feedback = generate_response([
        {"role": "system", "content": GENERATE_FEEDBACK},
        {"role": "user", "content": state.user_answer},
    ])
    return {
        "ai_feedback": feedback,
        "answers": [state.user_answer],
        "feedbacks": [feedback],
    }


graph = StateGraph(State)
graph.add_node("generate_question", generate_question)
graph.add_node("ask_for_answer", ask_for_answer)
graph.add_node("generate_feedback", generate_feedback)

graph.set_entry_point("generate_question")
graph.add_edge("generate_question", "ask_for_answer")
graph.add_conditional_edges(
    "ask_for_answer",
    should_stop,
    {"stop": END, "continue": "generate_feedback"},
)
graph.add_edge("generate_feedback", "generate_question")

app = graph.compile(checkpointer=MemorySaver())
