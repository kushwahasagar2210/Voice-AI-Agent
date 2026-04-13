"""
stt.py — Speech-to-Text via Groq Whisper API
Uses whisper-large-v3 for high accuracy on average hardware.
"""

from groq import Groq
import config


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio bytes to text using Groq's Whisper endpoint.

    Args:
        audio_bytes: Raw audio data as bytes.
        filename:    Original filename (used for MIME-type inference).

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If API key is missing.
        Exception:  On Groq API errors.
    """
    if not config.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file or enter it in the sidebar."
        )

    client = Groq(api_key=config.GROQ_API_KEY)

    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=config.STT_MODEL,
        response_format="text",
        language="en",          # set to None for auto-detect
        temperature=0.0,        # deterministic transcription
    )

    # Groq returns a plain string when response_format="text"
    return str(transcription).strip()
