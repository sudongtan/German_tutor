"""
Pydantic models for input validation and LLM structured output.

User input: UserTopic, UserAnswer, UserLevel — validated at the UI boundary.
LLM output: QuestionBatch, FeedbackResult — used with with_structured_output().
"""

from pydantic import BaseModel, Field, field_validator

VALID_LEVELS = {"A2", "B1", "B2", "C1"}


class UserLevel(BaseModel):
    level: str

    @field_validator("level")
    @classmethod
    def must_be_valid(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in VALID_LEVELS:
            raise ValueError("Level must be one of: A2, B1, B2, C1")
        return v


class QuestionBatch(BaseModel):
    questions: list[str]


class FeedbackResult(BaseModel):
    corrected_answer: str
    explanation: str
    score: int = Field(ge=1, le=10)


class UserTopic(BaseModel):
    topic: str = Field(min_length=1, max_length=100)

    @field_validator("topic")
    @classmethod
    def not_reserved(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Topic cannot be empty")
        if v.lower() == "stop":
            raise ValueError('"stop" is a reserved word, please enter a topic')
        return v


class GermanEquivalents(BaseModel):
    words: list[str]


class WordNuance(BaseModel):
    word: str
    frequency: str
    synonyms: list[str]
    nuance: str
    formality: str


class NuanceComparison(BaseModel):
    summary: str
    words: list[WordNuance]


class UserAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=500)

    @field_validator("answer")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Answer cannot be empty")
        return v
