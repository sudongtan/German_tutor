from unittest.mock import patch

from src.german_tutor.article_rag import (
    RagState,
    answer_follow_up,
    generate_answer,
    retrieve_chunks,
    route_follow_up,
)


def make_state(**kwargs) -> RagState:
    return RagState(**kwargs)


class TestRetrieveChunks:
    def test_returns_chunks(self):
        state = make_state(user_question="How do you express sadness?")
        with patch(
            "src.german_tutor.article_rag._store.search",
            return_value=["chunk one", "chunk two"],
        ):
            result = retrieve_chunks(state)
        assert result["chunks"] == ["chunk one", "chunk two"]

    def test_passes_question_to_search(self):
        state = make_state(user_question="How do you express sadness?")
        with patch(
            "src.german_tutor.article_rag._store.search",
            return_value=[],
        ) as mock:
            retrieve_chunks(state)
        mock.assert_called_once_with("How do you express sadness?")


class TestGenerateAnswer:
    def test_returns_answer(self):
        state = make_state(
            user_question="How do you say tired?",
            chunks=["müde means tired"],
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="Use 'müde' for tired.",
        ):
            result = generate_answer(state)
        assert result["answer"] == "Use 'müde' for tired."

    def test_includes_question_in_prompt(self):
        state = make_state(
            user_question="How do you say tired?",
            chunks=["müde means tired"],
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="...",
        ) as mock:
            generate_answer(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "How do you say tired?" in prompt

    def test_includes_chunks_in_prompt(self):
        state = make_state(
            user_question="...",
            chunks=["müde means tired", "erschöpft means exhausted"],
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="...",
        ) as mock:
            generate_answer(state)
        prompt = mock.call_args[0][0][0]["content"]
        assert "müde means tired" in prompt
        assert "erschöpft means exhausted" in prompt


class TestRouteFollowUp:
    def test_stop_returns_stop(self):
        assert route_follow_up(make_state(follow_up_question="stop")) == "stop"

    def test_stop_case_insensitive(self):
        assert route_follow_up(make_state(follow_up_question="STOP")) == "stop"

    def test_stop_strips_whitespace(self):
        assert route_follow_up(make_state(follow_up_question="  stop  ")) == "stop"

    def test_question_returns_answer(self):
        assert (
            route_follow_up(make_state(follow_up_question="Tell me more.")) == "answer"
        )


class TestAnswerFollowUp:
    def test_returns_answer(self):
        state = make_state(
            chunks=["müde means tired"],
            answer="Use müde.",
            follow_up_question="Any other words?",
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="Also erschöpft.",
        ):
            result = answer_follow_up(state)
        assert result["follow_up_answers"] == ["Also erschöpft."]

    def test_appends_to_follow_up_messages(self):
        state = make_state(
            chunks=["chunk"],
            answer="Initial answer.",
            follow_up_question="More?",
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="Yes, more.",
        ):
            result = answer_follow_up(state)
        msgs = result["follow_up_messages"]
        assert msgs[0].content == "More?"
        assert msgs[1].content == "Yes, more."

    def test_includes_previous_messages_in_context(self):
        from langchain_core.messages import AIMessage, HumanMessage

        previous = [HumanMessage(content="First?"), AIMessage(content="First answer.")]
        state = make_state(
            chunks=["chunk"],
            answer="Initial.",
            follow_up_question="Second?",
            follow_up_messages=previous,
        )
        with patch(
            "src.german_tutor.article_rag.generate_response",
            return_value="...",
        ) as mock:
            answer_follow_up(state)
        contents = [m.content for m in mock.call_args[0][0]]
        assert "First?" in contents
        assert "First answer." in contents
