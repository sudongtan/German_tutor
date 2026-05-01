import io

from openai import OpenAI

from src.german_tutor.config import settings


def transcribe_audio(audio_bytes: bytes) -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.webm"
    try:
        result = client.audio.transcriptions.create(
            model=settings.whisper_model,
            file=audio_file,
            language="de",
        )
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    return result.text.strip()
