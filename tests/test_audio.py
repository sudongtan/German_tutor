from unittest.mock import MagicMock, patch

import pytest

from src.german_tutor.audio import transcribe_audio


def make_openai_client(text):
    client = MagicMock()
    client.audio.transcriptions.create.return_value.text = text
    return client


class TestTranscribeAudio:
    def test_returns_transcript(self):
        with patch(
            "src.german_tutor.audio.OpenAI",
            return_value=make_openai_client("Ich bin müde"),
        ):
            result = transcribe_audio(b"fake-audio")
        assert result == "Ich bin müde"

    def test_strips_whitespace(self):
        with patch(
            "src.german_tutor.audio.OpenAI",
            return_value=make_openai_client("  Ich bin müde  "),
        ):
            result = transcribe_audio(b"fake-audio")
        assert result == "Ich bin müde"

    def test_passes_whisper_model(self):
        client = make_openai_client("Hallo")
        with patch("src.german_tutor.audio.OpenAI", return_value=client):
            transcribe_audio(b"fake-audio")
        call_kwargs = client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-1"

    def test_passes_german_language(self):
        client = make_openai_client("Hallo")
        with patch("src.german_tutor.audio.OpenAI", return_value=client):
            transcribe_audio(b"fake-audio")
        call_kwargs = client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "de"

    def test_raises_runtime_error_on_failure(self):
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = Exception("API error")
        with patch("src.german_tutor.audio.OpenAI", return_value=client):
            with pytest.raises(RuntimeError, match="Transcription failed"):
                transcribe_audio(b"fake-audio")

    def test_preserves_original_cause(self):
        client = MagicMock()
        original = Exception("network error")
        client.audio.transcriptions.create.side_effect = original
        with patch("src.german_tutor.audio.OpenAI", return_value=client):
            with pytest.raises(RuntimeError) as exc_info:
                transcribe_audio(b"fake-audio")
        assert exc_info.value.__cause__ is original
