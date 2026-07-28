import logging
import os
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from routers.trained_analysis import _parse_word_spans, _transcribe_free_speech
from services.gemma_alignment import refine_alignment_with_gemma
from services.text_signal_fusion import fuse_text_signals
from services.trained_ending_detector import TrainedEndingDetector
from utils.audio_utils import audio_array_to_wav_bytes, normalize_for_asr, trim_silence, webm_to_wav

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Arabic API wrapper"])


def _check_api_key(x_api_key: Optional[str]) -> None:
    expected = os.environ.get("ARABIC_WRAPPER_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def _summarize_findings(findings: list[dict]) -> dict:
    total = len(findings)
    correct = sum(1 for item in findings if item.get("status") == "correct")
    incorrect = sum(1 for item in findings if item.get("status") == "incorrect")
    uncertain = sum(1 for item in findings if item.get("status") == "uncertain")
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "accuracy": round(correct / total, 4) if total else None,
    }


def _normalize_mode(mode: str, has_reference: bool) -> str:
    value = (mode or "auto").strip().lower()
    if value not in {"auto", "light", "full"}:
        raise HTTPException(status_code=400, detail="mode must be one of: auto, light, full")
    if value == "auto":
        return "full" if has_reference else "light"
    if value == "full" and not has_reference:
        raise HTTPException(status_code=400, detail="full mode requires reference_text")
    return value


@router.get("/arabic-wrapper/health")
async def arabic_wrapper_health(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Health check for external apps that call the Arabic service over HTTP."""
    _check_api_key(x_api_key)
    return {
        "status": "ok",
        "service": "etib-arabic-wrapper",
        "version": "1.0",
        "modes": ["light", "full", "auto"],
        "configured": {
            "cohere": bool(os.environ.get("COHERE_API_KEY")),
            "groq": bool(os.environ.get("GROQ_API_KEY")),
            "gemma_endpoint": bool(os.environ.get("GEMMA_OPENAI_BASE_URL")),
            "wrapper_api_key_required": bool(os.environ.get("ARABIC_WRAPPER_API_KEY")),
        },
        "contract": {
            "method": "POST",
            "path": "/api/arabic-wrapper/evaluate",
            "content_type": "multipart/form-data",
            "fields": {
                "audio": "required file; webm, wav, m4a, mp3 accepted if ffmpeg can decode it",
                "reference_text": "optional in light mode, required in full mode",
                "mode": "auto, light, or full",
                "exercise_id": "optional string",
                "word_spans_json": "optional forced-alignment spans as JSON list",
            },
        },
    }


@router.post("/arabic-wrapper/evaluate")
async def evaluate_arabic_from_external_app(
    audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    mode: str = Form("auto"),
    exercise_id: Optional[str] = Form(None),
    word_spans_json: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Stable API wrapper for Joe's Flask deployment.

    The Flask app should call this endpoint through HTTP instead of importing
    this FastAPI code. That keeps both apps independent and keeps heavy Arabic
    models on the Arabic service only.
    """
    _check_api_key(x_api_key)
    reference = (reference_text or "").strip()
    mode_used = _normalize_mode(mode, bool(reference))

    raw_bytes = await audio.read()
    logger.info(
        "arabic wrapper request: filename=%s content_type=%s bytes=%s mode=%s has_reference=%s",
        audio.filename,
        audio.content_type,
        len(raw_bytes),
        mode_used,
        bool(reference),
    )
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
        wav_bytes = audio_array_to_wav_bytes(audio_array, sr)
    except Exception as exc:
        logger.warning("wrapper audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    try:
        transcript = await _transcribe_free_speech(wav_bytes, reference)
    except Exception as exc:
        logger.exception("wrapper transcription failed")
        transcript = {
            "provider": "none",
            "model": None,
            "fallback_used": False,
            "transcript": "",
            "diacritized_transcript": "",
            "tanween_notes": [],
            "error": str(exc),
        }

    analysis_response: dict = {}
    findings: list[dict] = []
    pronunciation_error: Optional[str] = None

    if mode_used == "full":
        try:
            word_spans = _parse_word_spans(word_spans_json)
            if not word_spans:
                word_spans = await TrainedEndingDetector.transcribe_word_timestamps(wav_bytes)
            if not word_spans:
                word_spans = TrainedEndingDetector.ctc_word_timestamps(audio_array, sr)
            refined_spans = await refine_alignment_with_gemma(reference, word_spans)
            if refined_spans:
                word_spans = refined_spans

            analysis_response = TrainedEndingDetector.analyze_sentence(
                audio_array,
                sr,
                reference,
                word_spans=word_spans,
            )
            analysis_response = await TrainedEndingDetector.apply_sentence_hypothesis_scoring(
                analysis_response,
                wav_bytes,
            )
            analysis_response["speech_transcript"] = transcript
            analysis_response = fuse_text_signals(analysis_response, transcript)
            findings = analysis_response.get("iraab_findings", [])
        except Exception as exc:
            logger.exception("wrapper pronunciation path failed")
            pronunciation_error = str(exc)

    content = {
        "ok": True,
        "service": "etib-arabic-wrapper",
        "mode_used": mode_used,
        "input": {
            "exercise_id": exercise_id,
            "has_reference_text": bool(reference),
            "audio_filename": audio.filename,
            "audio_content_type": audio.content_type,
        },
        "transcript": {
            "provider": transcript.get("provider"),
            "model": transcript.get("model"),
            "fallback_used": transcript.get("fallback_used"),
            "raw": transcript.get("transcript", ""),
            "diacritized": transcript.get("diacritized_transcript", ""),
            "tanween_notes": transcript.get("tanween_notes", []),
            "error": transcript.get("error"),
        },
        "pronunciation": {
            "available": mode_used == "full" and pronunciation_error is None,
            "summary": _summarize_findings(findings),
            "findings": findings,
            "error": pronunciation_error,
        },
        "delivery": analysis_response.get("delivery", {}),
        "model_provenance": analysis_response.get("model_provenance", {}),
        "integration_note": (
            "Call this HTTP endpoint from the Flask app. Do not merge the Flask "
            "backend with the Arabic FastAPI service or import its model code."
        ),
    }
    return JSONResponse(content=content)

