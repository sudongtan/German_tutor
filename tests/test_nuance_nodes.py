from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.german_tutor.nuance_comparison import (
    NuanceState,
    answer_follow_up,
    generate_equivalents,
    route_after_run,
    route_follow_up,
    structure_comparison,
)
from src.german_tutor.schemas import GermanEquivalents


def make_state(**kwargs) -> NuanceState:
    return NuanceState(**kwargs)


class TestGenerateEquivalents:
    def test_returns_equivalents(self):
        state = make_state(english_input="tired")
        batch = GermanEquivalents(words=["müde", "erschöpft"])
        with patch(
            "src.german_tutor.nuance_comparison.generate_structured_response",
            return_value=batch,
        ):
            result = generate_equivalents(state)
        assert result["equivalents"] == ["müde", "erschöpft"]

    def test_includes_english_in_prompt(self):
        state = make_state(english_input="tired")
        batch = GermanEquivalents(words=["müde"])
        with patch(
            "src.german_tutor.nuance_comparison.generate_structured_response",
            return_value=batch,
        ) as mock:
            generate_equivalents(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "tired" in prompt


class TestStructureComparison:
    def test_returns_formatted_comparison(self):
        from src.german_tutor.schemas import NuanceComparison, WordNuance

        structured = NuanceComparison(
            summary="müde is more common.",
            words=[
                WordNuance(
                    word="müde",
                    frequency="very common",
                    synonyms=["erschöpft"],
                    nuance="everyday tiredness",
                    formality="neutral",
                )
            ],
        )
        state = make_state(messages=[AIMessage(content="müde is more common...")])
        with patch(
            "src.german_tutor.nuance_comparison.generate_structured_response",
            return_value=structured,
        ):
            result = structure_comparison(state)
        assert "müde" in result["comparison"]
        assert "very common" in result["comparison"]
        assert "neutral" in result["comparison"]

    def test_passes_last_message_to_prompt(self):
        from src.german_tutor.schemas import NuanceComparison, WordNuance

        structured = NuanceComparison(
            summary="s",
            words=[
                WordNuance(
                    word="w", frequency="f", synonyms=[], nuance="n", formality="fo"
                )
            ],
        )
        state = make_state(messages=[AIMessage(content="the free-text comparison")])
        with patch(
            "src.german_tutor.nuance_comparison.generate_structured_response",
            return_value=structured,
        ) as mock:
            structure_comparison(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "the free-text comparison" in prompt


class TestRouteAfterRun:
    def test_with_tool_calls_returns_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "frequency_lookup",
                    "args": {"word": "müde"},
                    "id": "abc123",
                    "type": "tool_call",
                }
            ],
        )
        state = make_state(messages=[msg])
        assert route_after_run(state) == "tools"

    def test_without_tool_calls_returns_structure(self):
        state = make_state(messages=[AIMessage(content="Here is the comparison.")])
        assert route_after_run(state) == "structure"


class TestAnswerFollowUp:
    def test_returns_answer(self):
        state = make_state(
            english_input="tired",
            equivalents=["müde", "erschöpft"],
            comparison="müde is everyday...",
            user_question="Which one is more formal?",
        )
        with patch(
            "src.german_tutor.nuance_comparison.generate_response",
            return_value="erschöpft is more formal.",
        ):
            result = answer_follow_up(state)
        assert result["follow_up_answers"] == ["erschöpft is more formal."]

    def test_appends_question_and_answer_to_follow_up_messages(self):
        state = make_state(
            english_input="tired",
            equivalents=["müde"],
            comparison="müde is everyday...",
            user_question="Which one is more formal?",
        )
        with patch(
            "src.german_tutor.nuance_comparison.generate_response",
            return_value="erschöpft is more formal.",
        ):
            result = answer_follow_up(state)
        msgs = result["follow_up_messages"]
        assert msgs[0].content == "Which one is more formal?"
        assert msgs[1].content == "erschöpft is more formal."

    def test_includes_previous_messages_in_context(self):
        from langchain_core.messages import AIMessage as AI
        from langchain_core.messages import HumanMessage as Human

        previous = [Human(content="Which is more formal?"), AI(content="erschöpft.")]
        state = make_state(
            english_input="tired",
            equivalents=["müde"],
            comparison="müde is everyday...",
            user_question="Give me an example.",
            follow_up_messages=previous,
        )
        with patch(
            "src.german_tutor.nuance_comparison.generate_response",
            return_value="...",
        ) as mock:
            answer_follow_up(state)
        contents = [m.content for m in mock.call_args[0][0]]
        assert "Which is more formal?" in contents
        assert "erschöpft." in contents

    def test_includes_comparison_in_system_message(self):
        state = make_state(
            english_input="tired",
            equivalents=["müde"],
            comparison="müde is everyday...",
            user_question="Tell me more.",
        )
        with patch(
            "src.german_tutor.nuance_comparison.generate_response",
            return_value="...",
        ) as mock:
            answer_follow_up(state)
        assert "müde is everyday..." in mock.call_args[0][0][0].content


class TestRouteFollowUp:
    def test_stop_returns_stop(self):
        assert route_follow_up(make_state(user_question="stop")) == "stop"

    def test_stop_case_insensitive(self):
        assert route_follow_up(make_state(user_question="STOP")) == "stop"

    def test_stop_strips_whitespace(self):
        assert route_follow_up(make_state(user_question="  stop  ")) == "stop"

    def test_question_returns_answer(self):
        assert (
            route_follow_up(make_state(user_question="Which is more formal?"))
            == "answer"
        )
