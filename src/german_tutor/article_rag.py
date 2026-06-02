"""
LangGraph graph for RAG-based German language Q&A.

First invocation: invoke({"user_question": str}, config).
Subsequent turns: invoke(Command(resume=follow_up_question), config).
Exposes `rag_app` (compiled graph) and `RagState` for external use.

Indexing is handled separately via weaviate_client.index_article() —
it is not part of this graph.
"""

import operator
import sqlite3
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.german_tutor.config import settings
from src.german_tutor.llm import generate_response
from src.german_tutor.prompts import RAG_ANSWER, RAG_FOLLOW_UP
from src.german_tutor.stores import store as _store


class RagState(BaseModel):
    user_question: str = ""
    chunks: list[str] = Field(default_factory=list)
    answer: str = ""
    follow_up_question: str = ""
    follow_up_messages: Annotated[list, add_messages] = Field(default_factory=list)
    follow_up_answers: Annotated[list[str], operator.add] = Field(default_factory=list)


def retrieve_chunks(state: RagState):
    chunks = _store.search(state.user_question)
    return {"chunks": chunks}


def generate_answer(state: RagState):
    if not state.chunks:
        return {
            "answer": "No relevant content found. Try indexing more articles or rephrasing your question."
        }
    chunks_text = "\n\n---\n\n".join(state.chunks)
    answer = generate_response(
        [
            {
                "role": "system",
                "content": RAG_ANSWER.format(
                    chunks=chunks_text,
                    question=state.user_question,
                ),
            }
        ]
    )
    return {"answer": answer}


def ask_follow_up(state: RagState):
    follow_up = interrupt(state.answer)
    return {"follow_up_question": follow_up}


def route_follow_up(state: RagState) -> str:
    if state.follow_up_question.strip().lower() == "stop":
        return "stop"
    return "answer"


def answer_follow_up(state: RagState):
    chunks_text = "\n\n---\n\n".join(state.chunks)
    system = SystemMessage(
        content=RAG_FOLLOW_UP.format(chunks=chunks_text, answer=state.answer)
    )
    messages = (
        [system]
        + list(state.follow_up_messages)
        + [HumanMessage(content=state.follow_up_question)]
    )
    answer = generate_response(messages)
    return {
        "follow_up_answers": [answer],
        "follow_up_messages": [
            HumanMessage(content=state.follow_up_question),
            AIMessage(content=answer),
        ],
    }


rag_graph = StateGraph(RagState)
rag_graph.add_node("retrieve_chunks", retrieve_chunks)
rag_graph.add_node("generate_answer", generate_answer)
rag_graph.add_node("ask_follow_up", ask_follow_up)
rag_graph.add_node("answer_follow_up", answer_follow_up)

rag_graph.set_entry_point("retrieve_chunks")
rag_graph.add_edge("retrieve_chunks", "generate_answer")
rag_graph.add_edge("generate_answer", "ask_follow_up")
rag_graph.add_conditional_edges(
    "ask_follow_up",
    route_follow_up,
    {"stop": END, "answer": "answer_follow_up"},
)
rag_graph.add_edge("answer_follow_up", "ask_follow_up")

_conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
rag_app = rag_graph.compile(checkpointer=SqliteSaver(_conn))
