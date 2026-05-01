"""
LangGraph graph for German nuance comparison.

First invocation: invoke({"english_input": str}, config).
Subsequent turns: invoke(Command(resume=user_question), config).
Exposes `nuance_app` (compiled graph) and `NuanceState` for external use.

The comparison phase uses a ReAct loop: the LLM decides whether to call
frequency_lookup and/or synonyms_lookup before writing its final comparison.
"""

import operator
import sqlite3
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.german_tutor.config import LLM, settings
from src.german_tutor.dictionary import frequency_lookup, synonyms_lookup
from src.german_tutor.llm import generate_response, generate_structured_response
from src.german_tutor.prompts import (
    GENERATE_EQUIVALENTS,
    GENERATE_FOLLOW_UP_ANSWER,
    GENERATE_NUANCE_COMPARISON,
    STRUCTURE_COMPARISON,
)
from src.german_tutor.schemas import GermanEquivalents, NuanceComparison

_tools = [frequency_lookup, synonyms_lookup]
_llm_with_tools = LLM.bind_tools(_tools)


class NuanceState(BaseModel):
    english_input: str = ""
    equivalents: list[str] = Field(default_factory=list)
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    comparison: str = ""
    user_question: str = ""
    follow_up_messages: Annotated[list, add_messages] = Field(default_factory=list)
    follow_up_answers: Annotated[list[str], operator.add] = Field(default_factory=list)


def generate_equivalents(state: NuanceState):
    result = generate_structured_response(
        [
            {
                "role": "system",
                "content": GENERATE_EQUIVALENTS.format(english=state.english_input),
            }
        ],
        GermanEquivalents,
    )
    return {"equivalents": result.words}


def run_comparison(state: NuanceState):
    if not state.messages:
        system = SystemMessage(
            content=GENERATE_NUANCE_COMPARISON.format(
                english=state.english_input,
                equivalents=", ".join(state.equivalents),
            )
        )
        response = _llm_with_tools.invoke([system])
        return {"messages": [system, response]}
    response = _llm_with_tools.invoke(state.messages)
    return {"messages": [response]}


def route_after_run(state: NuanceState) -> str:
    last = state.messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "structure"


def _format_comparison(c: NuanceComparison) -> str:
    lines = [f"**{c.summary}**\n"]
    for w in c.words:
        lines.append(f"**{w.word}**")
        lines.append(f"- Frequency: {w.frequency}")
        lines.append(f"- Formality: {w.formality}")
        lines.append(f"- Synonyms: {', '.join(w.synonyms) if w.synonyms else '—'}")
        lines.append(f"- Nuance: {w.nuance}")
        lines.append("")
    return "\n".join(lines)


def structure_comparison(state: NuanceState):
    result = generate_structured_response(
        [
            {
                "role": "user",
                "content": STRUCTURE_COMPARISON.format(
                    comparison=state.messages[-1].content
                ),
            }
        ],
        NuanceComparison,
    )
    return {"comparison": _format_comparison(result)}


def ask_follow_up(state: NuanceState):
    user_question = interrupt(state.comparison)
    return {"user_question": user_question}


def route_follow_up(state: NuanceState) -> str:
    if state.user_question.strip().lower() == "stop":
        return "stop"
    return "answer"


def answer_follow_up(state: NuanceState):
    system = SystemMessage(
        content=GENERATE_FOLLOW_UP_ANSWER.format(
            english=state.english_input,
            equivalents=", ".join(state.equivalents),
            comparison=state.comparison,
        )
    )
    messages = (
        [system]
        + list(state.follow_up_messages)
        + [HumanMessage(content=state.user_question)]
    )
    answer = generate_response(messages)
    return {
        "follow_up_answers": [answer],
        "follow_up_messages": [
            HumanMessage(content=state.user_question),
            AIMessage(content=answer),
        ],
    }


nuance_graph = StateGraph(NuanceState)
nuance_graph.add_node("generate_equivalents", generate_equivalents)
nuance_graph.add_node("run_comparison", run_comparison)
nuance_graph.add_node("tools", ToolNode(_tools))
nuance_graph.add_node("structure_comparison", structure_comparison)
nuance_graph.add_node("ask_follow_up", ask_follow_up)
nuance_graph.add_node("answer_follow_up", answer_follow_up)

nuance_graph.set_entry_point("generate_equivalents")
nuance_graph.add_edge("generate_equivalents", "run_comparison")
nuance_graph.add_conditional_edges(
    "run_comparison",
    route_after_run,
    {"tools": "tools", "structure": "structure_comparison"},
)
nuance_graph.add_edge("tools", "run_comparison")
nuance_graph.add_edge("structure_comparison", "ask_follow_up")
nuance_graph.add_conditional_edges(
    "ask_follow_up",
    route_follow_up,
    {"stop": END, "answer": "answer_follow_up"},
)
nuance_graph.add_edge("answer_follow_up", "ask_follow_up")

_conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
nuance_app = nuance_graph.compile(checkpointer=SqliteSaver(_conn))
