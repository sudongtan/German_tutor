from unittest.mock import patch

from src.german_tutor.schemas import FeedbackResult, QuestionBatch
from src.german_tutor.tutor import (
    State,
    generate_examples,
    generate_expressions,
    generate_feedback,
    generate_question,
    route_input,
)


def make_state(**kwargs) -> State:
    return State(**kwargs)


def make_feedback(corrected="Ich bin müde.", explanation="Good job.", score=8):
    return FeedbackResult(
        corrected_answer=corrected, explanation=explanation, score=score
    )


class TestGenerateFeedback:
    def test_returns_corrected_answer(self):
        state = make_state(user_answer="Ich bin mude")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_feedback(corrected="Ich bin müde."),
        ):
            result = generate_feedback(state)
        assert result["feedback_corrected_answer"] == "Ich bin müde."

    def test_returns_explanation(self):
        state = make_state(user_answer="Ich bin mude")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_feedback(explanation="Missing umlaut on 'müde'."),
        ):
            result = generate_feedback(state)
        assert result["feedback_explanation"] == "Missing umlaut on 'müde'."

    def test_returns_score(self):
        state = make_state(user_answer="Ich bin müde")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_feedback(score=9),
        ):
            result = generate_feedback(state)
        assert result["feedback_score"] == 9

    def test_logs_answer(self):
        state = make_state(user_answer="Ich bin müde")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_feedback(),
        ):
            result = generate_feedback(state)
        assert result["answers"] == ["Ich bin müde"]

    def test_logs_feedback_as_dict(self):
        state = make_state(user_answer="Ich bin müde")
        fb = make_feedback(corrected="Ich bin müde.", explanation="Perfect.", score=10)
        with patch(
            "src.german_tutor.tutor.generate_structured_response", return_value=fb
        ):
            result = generate_feedback(state)
        assert result["feedbacks"] == [fb.model_dump()]

    def test_resets_moved_to_next(self):
        state = make_state(user_answer="Ich bin müde")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_feedback(),
        ):
            result = generate_feedback(state)
        assert result["moved_to_next"] is False


def make_batch(*questions):
    return QuestionBatch(questions=list(questions))


class TestGenerateQuestion:
    def test_returns_question(self):
        state = make_state(user_topic="weather")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ):
            result = generate_question(state)
        assert result["ai_question"] == "Wie ist das Wetter?"

    def test_appends_to_asked_questions(self):
        state = make_state(user_topic="weather")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ):
            result = generate_question(state)
        assert result["asked_questions"] == ["Wie ist das Wetter?"]

    def test_resets_hint(self):
        state = make_state(user_topic="weather", hint="some previous hint")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ):
            result = generate_question(state)
        assert result["hint"] == ""

    def test_sets_moved_to_next(self):
        state = make_state(user_topic="weather")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ):
            result = generate_question(state)
        assert result["moved_to_next"] is True

    def test_includes_avoid_in_prompt_when_previous_questions(self):
        state = make_state(
            user_topic="weather",
            user_level="B1",
            asked_questions=["Wie ist das Wetter?"],
        )
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Ist es kalt?"),
        ) as mock:
            generate_question(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Wie ist das Wetter?" in prompt

    def test_no_avoid_in_prompt_on_first_question(self):
        state = make_state(user_topic="weather", user_level="B1")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ) as mock:
            generate_question(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Do not ask" not in prompt

    def test_includes_level_in_prompt(self):
        state = make_state(user_topic="weather", user_level="C1")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Wie ist das Wetter?"),
        ) as mock:
            generate_question(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "C1" in prompt

    def test_pops_from_pool_without_llm_call(self):
        state = make_state(user_topic="weather", question_pool=["Frage 1", "Frage 2"])
        with patch("src.german_tutor.tutor.generate_structured_response") as mock:
            result = generate_question(state)
        mock.assert_not_called()
        assert result["ai_question"] == "Frage 1"

    def test_pool_decremented_after_pop(self):
        state = make_state(user_topic="weather", question_pool=["Frage 1", "Frage 2"])
        result = generate_question(state)
        assert result["question_pool"] == ["Frage 2"]

    def test_generates_batch_when_pool_empty(self):
        state = make_state(user_topic="weather")
        with patch(
            "src.german_tutor.tutor.generate_structured_response",
            return_value=make_batch("Frage 1", "Frage 2", "Frage 3"),
        ) as mock:
            result = generate_question(state)
        mock.assert_called_once()
        assert result["ai_question"] == "Frage 1"
        assert result["question_pool"] == ["Frage 2", "Frage 3"]


class TestGenerateExamples:
    def test_returns_hint(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="B1")
        with patch(
            "src.german_tutor.tutor.generate_response", return_value="Es ist sonnig."
        ):
            result = generate_examples(state)
        assert result["hint"] == "Es ist sonnig."

    def test_includes_question_in_prompt(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="B1")
        with patch(
            "src.german_tutor.tutor.generate_response", return_value="Es ist sonnig."
        ) as mock:
            generate_examples(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Wie ist das Wetter?" in prompt

    def test_includes_level_in_prompt(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="A2")
        with patch(
            "src.german_tutor.tutor.generate_response", return_value="Es ist sonnig."
        ) as mock:
            generate_examples(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "A2" in prompt


class TestGenerateExpressions:
    def test_returns_hint(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="B1")
        with patch(
            "src.german_tutor.tutor.generate_response", return_value="Das Wetter ist..."
        ):
            result = generate_expressions(state)
        assert result["hint"] == "Das Wetter ist..."

    def test_includes_question_in_prompt(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="B1")
        with patch(
            "src.german_tutor.tutor.generate_response",
            return_value="Das Wetter ist...",
        ) as mock:
            generate_expressions(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "Wie ist das Wetter?" in prompt

    def test_includes_level_in_prompt(self):
        state = make_state(ai_question="Wie ist das Wetter?", user_level="C1")
        with patch(
            "src.german_tutor.tutor.generate_response",
            return_value="Das Wetter ist...",
        ) as mock:
            generate_expressions(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "C1" in prompt


class TestRouteInput:
    def test_stop_returns_stop(self):
        assert route_input(make_state(user_answer="stop")) == "stop"

    def test_stop_case_insensitive(self):
        assert route_input(make_state(user_answer="STOP")) == "stop"

    def test_stop_strips_whitespace(self):
        assert route_input(make_state(user_answer="  stop  ")) == "stop"

    def test_1_returns_examples(self):
        assert route_input(make_state(user_answer="1")) == "examples"

    def test_2_returns_expressions(self):
        assert route_input(make_state(user_answer="2")) == "expressions"

    def test_next_returns_next(self):
        assert route_input(make_state(user_answer="next")) == "next"

    def test_next_case_insensitive(self):
        assert route_input(make_state(user_answer="NEXT")) == "next"

    def test_answer_returns_answer(self):
        assert route_input(make_state(user_answer="Ich bin müde")) == "answer"
