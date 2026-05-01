from unittest.mock import patch

import pytest

from src.german_tutor.tutor import (
    State,
    generate_feedback,
    generate_question,
    should_stop,
)


def make_state(**kwargs) -> State:
    return State(**kwargs)


class TestGenerateFeedback:
    def test_returns_feedback(self):
        state = make_state(user_answer="Ich bin müde")
        with patch("src.german_tutor.tutor.generate_response", return_value="Gut gemacht!"):
            result = generate_feedback(state)
        assert result["ai_feedback"] == "Gut gemacht!"

    def test_logs_answer(self):
        state = make_state(user_answer="Ich bin müde")
        with patch("src.german_tutor.tutor.generate_response", return_value="Gut gemacht!"):
            result = generate_feedback(state)
        assert result["answers"] == ["Ich bin müde"]

    def test_logs_feedback(self):
        state = make_state(user_answer="Ich bin müde")
        with patch("src.german_tutor.tutor.generate_response", return_value="Gut gemacht!"):
            result = generate_feedback(state)
        assert result["feedbacks"] == ["Gut gemacht!"]

    def test_uses_attribute_access(self):
        state = make_state(user_answer="Ich bin müde")
        with patch("src.german_tutor.tutor.generate_response", return_value="Gut!"):
            result = generate_feedback(state)
        assert "answers" in result and result["answers"] == ["Ich bin müde"]


class TestGenerateQuestion:
    def test_returns_question(self):
        state = make_state(user_topic="weather")
        with patch("src.german_tutor.tutor.generate_response", return_value="Wie ist das Wetter?"):
            result = generate_question(state)
        assert result["ai_question"] == "Wie ist das Wetter?"

    def test_appends_to_asked_questions(self):
        state = make_state(user_topic="weather")
        with patch("src.german_tutor.tutor.generate_response", return_value="Wie ist das Wetter?"):
            result = generate_question(state)
        assert result["asked_questions"] == ["Wie ist das Wetter?"]

    def test_includes_avoid_in_prompt_when_previous_questions(self):
        state = make_state(user_topic="weather", asked_questions=["Wie ist das Wetter?"])
        with patch("src.german_tutor.tutor.generate_response", return_value="Ist es kalt?") as mock:
            generate_question(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Wie ist das Wetter?" in prompt

    def test_no_avoid_in_prompt_on_first_question(self):
        state = make_state(user_topic="weather")
        with patch("src.german_tutor.tutor.generate_response", return_value="Wie ist das Wetter?") as mock:
            generate_question(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Do not ask" not in prompt


class TestShouldStop:
    def test_stop_returns_stop(self):
        assert should_stop(make_state(user_answer="stop")) == "stop"

    def test_stop_case_insensitive(self):
        assert should_stop(make_state(user_answer="STOP")) == "stop"

    def test_stop_strips_whitespace(self):
        assert should_stop(make_state(user_answer="  stop  ")) == "stop"

    def test_answer_returns_continue(self):
        assert should_stop(make_state(user_answer="Ich bin müde")) == "continue"
