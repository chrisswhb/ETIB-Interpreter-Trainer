import logging
import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from routers.speech_diacritize import _transcribe_arabic_with_groq
from services.cohere_transcriber import transcribe_arabic_with_cohere
from services.gemma_alignment import refine_alignment_with_gemma
from services.tanween_corrector import correct_tanween_with_llm
from services.text_signal_fusion import fuse_text_signals
from services.trained_ending_detector import TrainedEndingDetector
from utils.audio_utils import audio_array_to_wav_bytes, normalize_for_asr, trim_silence, webm_to_wav

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze-trained")
async def analyze_trained_audio(
    audio: UploadFile = File(...),
    reference_sentence: str = Form(...),
    exercise_id: Optional[str] = Form(None),
    word_spans_json: Optional[str] = Form(None),
):
    raw_bytes = await audio.read()
    logger.info(
        "trained analysis request: filename=%s content_type=%s bytes=%s",
        audio.filename,
        audio.content_type,
        len(raw_bytes),
    )
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
    except Exception as exc:
        logger.warning("audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    try:
        wav_bytes = audio_array_to_wav_bytes(audio_array, sr)
        speech_transcript = await _transcribe_free_speech(wav_bytes, reference_sentence)
        word_spans = _parse_word_spans(word_spans_json)
        if not word_spans:
            word_spans = await TrainedEndingDetector.transcribe_word_timestamps(wav_bytes)
        if not word_spans:
            word_spans = TrainedEndingDetector.ctc_word_timestamps(audio_array, sr)
        refined_spans = await refine_alignment_with_gemma(reference_sentence, word_spans)
        if refined_spans:
            word_spans = refined_spans
        response = TrainedEndingDetector.analyze_sentence(
            audio_array,
            sr,
            reference_sentence,
            word_spans=word_spans,
        )
        response = await TrainedEndingDetector.apply_sentence_hypothesis_scoring(
            response,
            wav_bytes,
        )
        response["speech_transcript"] = speech_transcript
        response = fuse_text_signals(response, speech_transcript)
    except Exception as exc:
        logger.exception("trained ending analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response["exercise_id"] = exercise_id
    return JSONResponse(content=response)


async def _transcribe_free_speech(wav_bytes: bytes, reference_sentence: str) -> dict:
    """Best available free-speech Arabic transcript for ETIB.

    This is intentionally separate from final-haraka acoustic verification:
    ASR gives the words the learner likely said, then the LLM creates a
    diacritized text hypothesis. The acoustic classifier still decides final
    haraka where alignment is available.
    """
    cohere_result = await transcribe_arabic_with_cohere(wav_bytes)
    transcript = cohere_result.get("text", "")
    provider = "cohere"
    fallback_used = False
    error = cohere_result.get("error")

    if not transcript:
        try:
            transcript = await _transcribe_arabic_with_groq(wav_bytes)
            provider = "groq_whisper_fallback"
            fallback_used = True
        except Exception as exc:
            error = f"{error or 'Cohere returned no transcript'}; Groq fallback failed: {exc}"

    correction = await correct_tanween_with_llm(transcript, reference_sentence) if transcript else {
        "diacritized_text": "",
        "tanween_notes": [],
        "error": error or "empty ASR transcript",
    }
    return {
        "provider": provider,
        "model": cohere_result.get("model") if provider == "cohere" else "whisper-large-v3",
        "fallback_used": fallback_used,
        "transcript": transcript,
        "diacritized_transcript": correction.get("diacritized_text", transcript),
        "tanween_notes": correction.get("tanween_notes", []),
        "correction_provider": correction.get("provider"),
        "error": error or correction.get("error"),
        "important_note": (
            "ASR transcript/tashkeel is a text hypothesis. Final pronunciation "
            "verification still depends on acoustic alignment and classification."
        ),
    }


def _parse_word_spans(value: Optional[str]) -> list[dict]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    spans = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip()
        start = item.get("start")
        end = item.get("end")
        if word and start is not None and end is not None:
            spans.append({
                "word": word,
                "start": float(start),
                "end": float(end),
                "source": str(item.get("source") or "provided_forced_alignment"),
            })
    return spans


@router.post("/analyze-guided-word")
async def analyze_guided_word_audio(
    audio: UploadFile = File(...),
    reference_word: str = Form(...),
    word_index: int = Form(0),
    exercise_id: Optional[str] = Form(None),
):
    """Analyze one prompted Arabic word.

    This endpoint intentionally avoids ASR and sentence alignment. The frontend
    already tells us which word the learner read, so the classifier receives one
    isolated word recording and decides only its final ending.
    """
    raw_bytes = await audio.read()
    logger.info(
        "guided word request: filename=%s content_type=%s bytes=%s word=%s",
        audio.filename,
        audio.content_type,
        len(raw_bytes),
        reference_word,
    )
    try:
        if len(raw_bytes) < 512:
            raise RuntimeError("the browser sent an empty recording; record again for at least one second")
        audio_array, sr = webm_to_wav(raw_bytes)
        audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
    except Exception as exc:
        logger.warning("guided word audio conversion failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {exc}") from exc

    try:
        response = TrainedEndingDetector.analyze_sentence(
            audio_array,
            sr,
            reference_word,
            word_spans=[],
        )
        for finding in response.get("iraab_findings", []):
            finding["word_index"] = word_index
            finding.setdefault("_debug", {})["guided_word_mode"] = True
        response["exercise_id"] = exercise_id
        response["guided_word_mode"] = True
        return JSONResponse(content=response)
    except Exception as exc:
        logger.exception("guided word analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
