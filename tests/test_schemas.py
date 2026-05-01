import pytest
from pydantic import ValidationError

from src.german_tutor.schemas import UserAnswer, UserTopic


class TestUserTopic:
    def test_valid_topic(self):
        t = UserTopic(topic="summer")
        assert t.topic == "summer"

    def test_strips_whitespace(self):
        t = UserTopic(topic="  food  ")
        assert t.topic == "food"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            UserTopic(topic="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            UserTopic(topic="   ")

    def test_stop_reserved_word_rejected(self):
        with pytest.raises(ValidationError, match="reserved word"):
            UserTopic(topic="stop")

    def test_stop_case_insensitive(self):
        with pytest.raises(ValidationError, match="reserved word"):
            UserTopic(topic="STOP")

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            UserTopic(topic="a" * 101)

    def test_max_length_accepted(self):
        t = UserTopic(topic="a" * 100)
        assert len(t.topic) == 100


class TestUserAnswer:
    def test_valid_answer(self):
        a = UserAnswer(answer="Ich bin müde")
        assert a.answer == "Ich bin müde"

    def test_strips_whitespace(self):
        a = UserAnswer(answer="  gut  ")
        assert a.answer == "gut"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            UserAnswer(answer="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValidationError):
            UserAnswer(answer="   ")

    def test_stop_is_valid_answer(self):
        a = UserAnswer(answer="stop")
        assert a.answer == "stop"

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            UserAnswer(answer="a" * 501)

    def test_max_length_accepted(self):
        a = UserAnswer(answer="a" * 500)
        assert len(a.answer) == 500
