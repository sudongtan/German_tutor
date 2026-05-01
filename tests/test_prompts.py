from src.german_tutor.prompts import (
    AVOID_REPEAT,
    GENERATE_FEEDBACK,
    GENERATE_QUESTIONS_BATCH,
)


def test_generate_questions_batch_has_topic_placeholder():
    assert "{topic}" in GENERATE_QUESTIONS_BATCH


def test_generate_questions_batch_has_level_placeholder():
    assert "{level}" in GENERATE_QUESTIONS_BATCH


def test_generate_questions_batch_has_avoid_placeholder():
    assert "{avoid}" in GENERATE_QUESTIONS_BATCH


def test_generate_questions_batch_formats_correctly():
    result = GENERATE_QUESTIONS_BATCH.format(topic="weather", level="B1", avoid="")
    assert "weather" in result
    assert "B1" in result


def test_generate_feedback_has_level_placeholder():
    assert "{level}" in GENERATE_FEEDBACK


def test_avoid_repeat_has_previous_placeholder():
    assert "{previous}" in AVOID_REPEAT


def test_avoid_repeat_formats_correctly():
    result = AVOID_REPEAT.format(previous=["Was ist das?"])
    assert "Was ist das?" in result


def test_generate_feedback_is_non_empty():
    assert GENERATE_FEEDBACK.strip()
