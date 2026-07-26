"""Cohere Arabic ASR adapter.

Uses Cohere Transcribe Arabic when COHERE_API_KEY is configured. This gives
the project a cleaner ASR baseline than Whisper for Arabic word transcription,
but Cohere's public docs still describe the output as text, not acoustic
tashkeel/tanween verification.
"""

import logging
import os

logger = logging.getLogger(__name__)

COHERE_TRANSCRIBE_URL = "https://api.cohere.com/v2/audio/transcriptions"
COHERE_ARABIC_MODEL = os.environ.get("COHERE_ARABIC_ASR_MODEL", "cohere-transcribe-arabic-07-2026")


async def transcribe_arabic_with_cohere(wav_bytes: bytes) -> dict:
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        return {
            "error": "COHERE_API_KEY is not configured",
            "text": "",
            "model": COHERE_ARABIC_MODEL,
            "provider": "cohere",
        }

    try:
        import httpx
    except Exception as exc:
        return {
            "error": f"httpx is required for Cohere transcription: {exc}",
            "text": "",
            "model": COHERE_ARABIC_MODEL,
            "provider": "cohere",
        }

    multipart = [
        ("model", (None, COHERE_ARABIC_MODEL)),
        ("language", (None, "ar")),
        ("temperature", (None, "0")),
        ("file", ("recording.wav", wav_bytes, "audio/wav")),
    ]
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                COHERE_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files=multipart,
            )
        if resp.status_code != 200:
            logger.warning("Cohere Arabic ASR failed %s: %s", resp.status_code, resp.text[:300])
            return {
                "error": f"Cohere transcription failed: {resp.status_code}",
                "text": "",
                "model": COHERE_ARABIC_MODEL,
                "provider": "cohere",
            }
        payload = resp.json()
        return {
            "text": (payload.get("text") or "").strip(),
            "model": COHERE_ARABIC_MODEL,
            "provider": "cohere",
        }
    except Exception as exc:
        logger.exception("Cohere Arabic ASR request failed")
        return {
            "error": str(exc),
            "text": "",
            "model": COHERE_ARABIC_MODEL,
            "provider": "cohere",
        }
