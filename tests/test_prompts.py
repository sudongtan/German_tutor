from src.german_tutor.prompts import AVOID_REPEAT, GENERATE_FEEDBACK, GENERATE_QUESTION


def test_generate_question_has_topic_placeholder():
    assert "{topic}" in GENERATE_QUESTION


def test_generate_question_has_avoid_placeholder():
    assert "{avoid}" in GENERATE_QUESTION


def test_generate_question_formats_correctly():
    result = GENERATE_QUESTION.format(topic="weather", avoid="")
    assert "weather" in result


def test_avoid_repeat_has_previous_placeholder():
    assert "{previous}" in AVOID_REPEAT


def test_avoid_repeat_formats_correctly():
    result = AVOID_REPEAT.format(previous=["Was ist das?"])
    assert "Was ist das?" in result


def test_generate_feedback_is_non_empty():
    assert GENERATE_FEEDBACK.strip()
