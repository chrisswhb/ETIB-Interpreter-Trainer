"""Speech-to-diacritized-Arabic hypothesis endpoint.

This is a practical substitute for unreleased CATT-Whisper weights:
Whisper produces the Arabic transcript, then the existing Arabic
diacritizer adds tashkeel. The result is useful for display, but it is not
acoustic proof that the learner pronounced each final haraka correctly.
"""

import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.diacritizer import diacritize_arabic_text
from utils.audio_utils import audio_array_to_wav_bytes, normalize_for_asr, trim_silence, webm_to_wav

logger = logging.getLogger(__name__)
router = APIRouter()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3"


async def _transcribe_arabic_with_groq(wav_bytes: bytes) -> str:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for Groq transcription") from exc

    files = {
        "file": ("recording.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "model": WHISPER_MODEL,
        "language": "ar",
        "response_format": "json",
        "temperature": "0",
    }
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{GROQ_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
        )

    if resp.status_code != 200:
        logger.warning("Groq Arabic transcription failed %s: %s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"Groq transcription failed: {resp.status_code}")

    payload = resp.json()
    return (payload.get("text") or "").strip()


@router.post("/speech-diacritize")
async def speech_diacritize(audio: UploadFile = File(...)):
    raw_bytes = await audio.read()
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
        wav_bytes = audio_array_to_wav_bytes(audio_array, sr)
    except Exception as exc:
        logger.warning("speech diacritization audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    try:
        transcript = await _transcribe_arabic_with_groq(wav_bytes)
        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript")

        diacritized = await diacritize_arabic_text(transcript)
        return JSONResponse(
            content={
                "transcript": transcript,
                "diacritized_text": diacritized.get("diacritized_text", transcript),
                "diacritizer_model": diacritized.get("model"),
                "method": "whisper-large-v3_asr_plus_llm_diacritizer",
                "warning": (
                    "This is a diacritized transcript hypothesis. It does not prove "
                    "the learner pronounced each final haraka acoustically."
                ),
            }
        )
    except Exception as exc:
        logger.exception("speech diacritization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
