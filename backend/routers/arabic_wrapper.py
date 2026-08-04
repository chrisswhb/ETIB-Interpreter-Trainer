import logging
import os
import re
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

_FINAL_MARKS = {
    "fatha": "\u064e",
    "damma": "\u064f",
    "kasra": "\u0650",
    "tanwin_fath": "\u064b",
    "tanwin_damm": "\u064c",
    "tanwin_kasr": "\u064d",
    "none": "",
}
_DIACRITIC_RE = re.compile(r"[\u064b-\u0652]+$")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _strip_final_marks(word: str) -> str:
    return _DIACRITIC_RE.sub("", word or "")


def _split_token(token: str) -> tuple[str, str, str]:
    match = re.match(r"^([^\u0600-\u06FF]*)(.*?)([^\u0600-\u06FF]*)$", token or "")
    if not match:
        return "", token or "", ""
    return match.group(1), match.group(2), match.group(3)


def _pseudo_reference_for_detection(text: str) -> str:
    """Ensure every Arabic token has a final mark so the acoustic detector targets it.

    For free speech there is no gold reference. We add a neutral dummy fatha only
    to make analyze_sentence classify every word ending; the final display uses
    detected_ending, not this dummy expected label.
    """
    words = []
    for token in (text or "").split():
        prefix, core, suffix = _split_token(token)
        if not core or not _ARABIC_RE.search(core):
            words.append(token)
            continue
        marked = core if re.search(r"[\u064b-\u0652]$", core) else f"{core}\u064e"
        words.append(f"{prefix}{marked}{suffix}")
    return " ".join(words)


def _spoken_harakat_text(base_text: str, findings: list[dict], min_confidence: float = 0.35) -> str:
    tokens = (base_text or "").split()
    if not tokens:
        return ""
    by_index = {int(item.get("word_index", -1)): item for item in findings if item.get("word_index") is not None}
    output = []
    for idx, token in enumerate(tokens):
        prefix, core, suffix = _split_token(token)
        item = by_index.get(idx)
        if not item or not core or not _ARABIC_RE.search(core):
            output.append(token)
            continue
        detected = str(item.get("detected_ending") or "none")
        confidence = float(item.get("detection_confidence") or 0.0)
        mark = _FINAL_MARKS.get(detected, "") if confidence >= min_confidence else ""
        output.append(f"{prefix}{_strip_final_marks(core)}{mark}{suffix}")
    return " ".join(output)


def _summarize_spoken_findings(findings: list[dict]) -> dict:
    total = len(findings)
    detected = sum(
        1 for item in findings
        if item.get("detected_ending") not in {None, "none"} and float(item.get("detection_confidence") or 0.0) >= 0.35
    )
    uncertain = sum(1 for item in findings if float(item.get("detection_confidence") or 0.0) < 0.35)
    return {"total_words": total, "detected_endings": detected, "uncertain": uncertain}

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
        "modes": ["light", "full", "auto", "spoken_harakat"],
        "configured": {
            "cohere": bool(os.environ.get("COHERE_API_KEY")),
            "groq": bool(os.environ.get("GROQ_API_KEY")),
            "gemma_endpoint": bool(os.environ.get("GEMMA_OPENAI_BASE_URL")),
            "wrapper_api_key_required": bool(os.environ.get("ARABIC_WRAPPER_API_KEY")),
        },
        "contract": {
            "method": "POST",
            "path": "/api/arabic-wrapper/evaluate or /api/arabic-wrapper/transcribe-harakat",
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


@router.post("/arabic-wrapper/transcribe-harakat")
async def transcribe_spoken_harakat(
    audio: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Free-speech Arabic transcription plus acoustic final-haraka detection.

    This endpoint is different from light mode: Cohere/Groq only provide rough
    words. Then the trained acoustic classifier listens to each word ending and
    reconstructs the final haraka/tanween it detected from the audio.
    """
    _check_api_key(x_api_key)
    raw_bytes = await audio.read()
    logger.info(
        "spoken harakat request: filename=%s content_type=%s bytes=%s",
        audio.filename,
        audio.content_type,
        len(raw_bytes),
    )
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
        wav_bytes = audio_array_to_wav_bytes(audio_array, sr)
    except Exception as exc:
        logger.warning("spoken harakat audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    try:
        transcript = await _transcribe_free_speech(wav_bytes, "")
    except Exception as exc:
        logger.exception("spoken harakat transcription failed")
        transcript = {
            "provider": "none",
            "model": None,
            "fallback_used": False,
            "transcript": "",
            "diacritized_transcript": "",
            "tanween_notes": [],
            "error": str(exc),
        }

    base_text = (transcript.get("transcript") or "").strip()
    text_hypothesis = (transcript.get("diacritized_transcript") or base_text).strip()
    pseudo_reference = _pseudo_reference_for_detection(text_hypothesis or base_text)
    findings: list[dict] = []
    acoustic_error: Optional[str] = None
    word_spans: list[dict] = []

    if pseudo_reference:
        try:
            word_spans = await TrainedEndingDetector.transcribe_word_timestamps(wav_bytes)
            if not word_spans:
                word_spans = TrainedEndingDetector.ctc_word_timestamps(audio_array, sr)
            refined_spans = await refine_alignment_with_gemma(pseudo_reference, word_spans)
            if refined_spans:
                word_spans = refined_spans
            acoustic_response = TrainedEndingDetector.analyze_sentence(
                audio_array,
                sr,
                pseudo_reference,
                word_spans=word_spans,
            )
            findings = acoustic_response.get("iraab_findings", [])
            for item in findings:
                item["status"] = "detected" if item.get("detected_ending") != "none" else "no_final_mark_detected"
                item["expected_ending"] = None
                item["explanation"] = "Free speech mode: this is the final ending detected acoustically, not a grammar-correctness comparison."
                item.setdefault("_debug", {})["free_speech_pseudo_reference"] = True
        except Exception as exc:
            logger.exception("spoken harakat acoustic detection failed")
            acoustic_error = str(exc)

    spoken_text = _spoken_harakat_text(text_hypothesis or base_text, findings)
    return JSONResponse(content={
        "ok": True,
        "service": "etib-arabic-wrapper",
        "mode_used": "spoken_harakat",
        "input": {
            "audio_filename": audio.filename,
            "audio_content_type": audio.content_type,
        },
        "transcript": {
            "provider": transcript.get("provider"),
            "model": transcript.get("model"),
            "fallback_used": transcript.get("fallback_used"),
            "raw": base_text,
            "diacritized": text_hypothesis,
            "spoken_harakat": spoken_text or text_hypothesis or base_text,
            "tanween_notes": transcript.get("tanween_notes", []),
            "error": transcript.get("error"),
        },
        "spoken_endings": {
            "available": bool(findings) and acoustic_error is None,
            "summary": _summarize_spoken_findings(findings),
            "findings": findings,
            "word_spans": word_spans,
            "error": acoustic_error,
        },
        "important_note": (
            "This is unrestricted speech mode. ASR supplies rough words; the trained acoustic "
            "classifier supplies the final haraka/tanween per aligned word. It is not a grammar answer key."
        ),
    })
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

