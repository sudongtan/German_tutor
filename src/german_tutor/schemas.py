from pydantic import BaseModel, Field, field_validator


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


class UserAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=500)

    @field_validator("answer")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Answer cannot be empty")
        return v
