import pytest
from pydantic import ValidationError

from src.german_tutor.schemas import UserAnswer, UserLevel, UserTopic


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


class TestUserLevel:
    def test_valid_levels(self):
        for level in ["A2", "B1", "B2", "C1"]:
            assert UserLevel(level=level).level == level

    def test_normalises_to_uppercase(self):
        assert UserLevel(level="b1").level == "B1"

    def test_strips_whitespace(self):
        assert UserLevel(level="  B2  ").level == "B2"

    def test_invalid_level_rejected(self):
        with pytest.raises(ValidationError, match="must be one of"):
            UserLevel(level="A1")

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            UserLevel(level="")

    def test_c2_rejected(self):
        with pytest.raises(ValidationError):
            UserLevel(level="C2")


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
