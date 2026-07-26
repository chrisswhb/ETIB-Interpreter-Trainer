"""Expert-recommended Arabic ASR -> tanween correction pipeline."""

import logging
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from routers.speech_diacritize import _transcribe_arabic_with_groq
from services.cohere_transcriber import transcribe_arabic_with_cohere
from services.tanween_corrector import correct_tanween_with_llm
from utils.audio_utils import audio_array_to_wav_bytes, normalize_for_asr, trim_silence, webm_to_wav

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/expert-pipeline-status")
async def expert_pipeline_status():
    return {
        "cohere_configured": bool(os.environ.get("COHERE_API_KEY")),
        "gemma_endpoint_configured": bool(os.environ.get("GEMMA_OPENAI_BASE_URL") and os.environ.get("GEMMA_API_KEY")),
        "groq_fallback_configured": bool(os.environ.get("GROQ_API_KEY")),
        "cohere_model": os.environ.get("COHERE_ARABIC_ASR_MODEL", "cohere-transcribe-arabic-07-2026"),
        "gemma_model": os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12B-it"),
    }


@router.post("/expert-tanween-pipeline")
async def expert_tanween_pipeline(
    audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
):
    """Run the proposed ASR -> LLM tanween correction experiment."""
    raw_bytes = await audio.read()
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
        wav_bytes = audio_array_to_wav_bytes(audio_array, sr)
    except Exception as exc:
        logger.warning("expert pipeline audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    cohere_result = await transcribe_arabic_with_cohere(wav_bytes)
    transcript = cohere_result.get("text", "")
    asr_provider = "cohere"
    fallback_used = False

    if not transcript:
        try:
            transcript = await _transcribe_arabic_with_groq(wav_bytes)
            asr_provider = "groq_whisper_fallback"
            fallback_used = True
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Cohere ASR failed and Groq fallback failed: {cohere_result.get('error')}; {exc}",
            ) from exc

    correction = await correct_tanween_with_llm(transcript, reference_text)
    return JSONResponse(
        content={
            "asr": {
                "provider": asr_provider,
                "model": cohere_result.get("model") if asr_provider == "cohere" else "whisper-large-v3",
                "fallback_used": fallback_used,
                "transcript": transcript,
                "cohere_error": cohere_result.get("error"),
            },
            "tanween_correction": correction,
            "reference_text": reference_text,
            "method": "cohere_transcribe_arabic_then_gemma_style_tanween_correction",
            "important_note": (
                "This tests whether ASR plus LLM correction can generate useful tanween/tashkeel. "
                "It is not yet acoustic proof of the learner's pronounced final haraka."
            ),
        }
    )
