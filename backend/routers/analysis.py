"""
analysis.py — POST /api/analyze

Accepts audio + exercise metadata, runs the full Layer 2 pipeline:
  1. WebM → WAV conversion
  2. wav2vec2 ensemble transcription (Signal 2)
  3. Competing Whisper hypothesis scoring (Signal 1)
  4. LLR logit analysis (Signal 3)
  5. Ensemble voting with confidence gating (Dr's addition)
  6. Silero VAD delivery metrics (Dr's addition)

Returns the layer2_delivery JSON object as specified by Dr.
"""

import logging
import asyncio
import re
import numpy as np
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from services.model_loader import ModelLoader
from services.acoustic_detector import (
    build_hypothesis_sentence,
    score_hypothesis_whisper,
    whisper_score_to_weight,
    wav2vec2_ensemble_vote,
    ctc_word_spans_from_logits,
    select_ctc_word_span,
    compute_llr_from_logits,
    merge_llr_scores,
    combine_signals,
    get_expected_ending_from_reference,
    HARAKA_ARABIZI,
)
from services.delivery_metrics import compute_delivery_metrics
from utils.audio_utils import webm_to_wav, audio_array_to_wav_bytes, trim_silence, normalize_for_asr
from utils.arabizi import arabic_to_arabizi, extract_ending, arabizi_to_haraka_label

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    reference_sentence: str = Form(...),    # fully diacritized Arabic
    focus_words: str = Form(...),           # JSON array of focus word strings
    exercise_id: Optional[str] = Form(None),
    feedback_language: str = Form("ar"),
):
    """
    Full i'rab detection pipeline for one exercise recording.
    Returns layer2_delivery JSON object.
    """
    import json

    # ── Parse inputs ──────────────────────────────────────────────────────────
    try:
        focus_word_list: list[str] = json.loads(focus_words)
    except json.JSONDecodeError:
        focus_word_list = [focus_words]

    # ── Audio preprocessing ───────────────────────────────────────────────────
    raw_bytes = await audio.read()
    try:
        audio_array, sr = webm_to_wav(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")
    original_duration_ms = int(len(audio_array) / sr * 1000) if sr else 0
    audio_array = normalize_for_asr(trim_silence(audio_array, threshold=0.008))
    wav_bytes = audio_array_to_wav_bytes(audio_array, sr)

    # ── wav2vec2 ensemble transcription ───────────────────────────────────────
    ensemble = ModelLoader.get_wav2vec2_ensemble()
    ensemble_results: list[tuple[Optional[str], Optional[np.ndarray]]] = []
    vocab_results: list[list[str]] = []
    span_results: list[list[dict]] = []

    for model, processor in ensemble:
        if model is None or processor is None:
            ensemble_results.append((None, None))
            vocab_results.append([])
            span_results.append([])
            continue
        try:
            import torch
            inputs = processor(
                audio_array, sampling_rate=sr, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)

            logits   = outputs.logits[0].numpy()
            pred_ids = np.argmax(logits, axis=-1)
            transcription = processor.batch_decode([pred_ids])[0]
            ensemble_results.append((transcription, logits))
            vocab = processor.tokenizer.get_vocab()
            vocab_list = [None] * len(vocab)
            for ch, idx in vocab.items():
                if idx < len(vocab_list):
                    vocab_list[idx] = ch
            vocab_results.append(vocab_list)
            span_results.append(ctc_word_spans_from_logits(logits, vocab_list))
        except Exception as e:
            logger.warning(f"wav2vec2 inference failed: {e}")
            ensemble_results.append((None, None))
            vocab_results.append([])
            span_results.append([])

    # Build a combined transcript from the ensemble (majority text)
    all_texts = [t for t, _ in ensemble_results if t]
    combined_transcript = all_texts[0] if all_texts else ""
    transcript_words    = combined_transcript.split()

    # ── Per-word analysis ─────────────────────────────────────────────────────
    # The exercise JSON highlights focus words, but students can make i'rab
    # mistakes elsewhere in the sentence. Analyze every reference word that has
    # an explicit final ending, and skip particles/words with no final haraka.
    reference_words = [_clean_reference_word(w) for w in reference_sentence.split()]
    analysis_targets = []
    for ref_idx, ref_word in enumerate(reference_words):
        expected = get_expected_ending_from_reference(ref_word)
        if expected != "none":
            analysis_targets.append({
                "word_index": ref_idx,
                "word": ref_word,
                "expected": expected,
                "is_focus": ref_word in focus_word_list,
            })

    iraab_findings = []

    for target in analysis_targets:
        word_idx = target["word_index"]
        focus_word = target["word"]
        ref_arabizi   = arabic_to_arabizi(focus_word)
        expected_haraka = target["expected"]

        # ── Signal 1: Competing Whisper hypothesis scoring ────────────────────
        # Generate 3 hypotheses (one per possible ending)
        candidate_harakaat = _candidate_harakaat(expected_haraka)
        whisper_scores: dict[str, float] = {}

        hypothesis_tasks = []
        for haraka in candidate_harakaat:
            hyp_sentence = build_hypothesis_sentence(reference_sentence, focus_word, haraka)
            hypothesis_tasks.append((haraka, hyp_sentence))

        # Run Whisper API calls concurrently
        async def _score(haraka, hyp):
            score = await score_hypothesis_whisper(wav_bytes, hyp)
            return haraka, score

        scored = await asyncio.gather(*[_score(h, s) for h, s in hypothesis_tasks])
        for haraka, score in scored:
            whisper_scores[haraka] = score

        # Pick best Whisper hypothesis
        if any(v > -float("inf") for v in whisper_scores.values()):
            sorted_scores  = sorted(whisper_scores.items(), key=lambda x: x[1], reverse=True)
            best_h, best_s = sorted_scores[0]
            sec_s          = sorted_scores[1][1] if len(sorted_scores) > 1 else best_s - 1
            whisper_result = {
                "haraka": best_h,
                "weight": whisper_score_to_weight(best_s, sec_s),
                "scores": {k: round(v, 4) for k, v in whisper_scores.items()},
            }
        else:
            whisper_result = {"haraka": None, "weight": 0}

        # ── Signal 2: wav2vec2 ensemble vote ──────────────────────────────────
        wav2vec2_result = wav2vec2_ensemble_vote(
            ensemble_results,
            ref_arabizi,
            word_index=word_idx,
            reference_word=focus_word,
            expected_haraka=expected_haraka,
        )

        # ── Signal 3: word-aware LLR logit analysis ──────────────────────────
        # Prefer CTC-derived word spans. If span matching fails, fall back to
        # the older rough sentence-fraction segment instead of dropping LLR.
        llr_scores_by_model = []
        matched_spans = []
        for (_, logits), vocab_list, spans in zip(ensemble_results, vocab_results, span_results):
            if logits is not None:
                try:
                    aligned_span = select_ctc_word_span(focus_word, spans, word_idx)
                    if aligned_span:
                        aligned_span["frame_count"] = len(logits)
                        matched_spans.append(aligned_span)
                    llr_scores_by_model.append(
                        compute_llr_from_logits(
                            logits,
                            vocab_list,
                            word_index=word_idx,
                            total_words=len(reference_words),
                            frame_start=aligned_span.get("start_frame") if aligned_span else None,
                            frame_end=aligned_span.get("end_frame") if aligned_span else None,
                        )
                    )
                except Exception as e:
                    logger.debug(f"LLR computation skipped: {e}")
        llr_result = merge_llr_scores(llr_scores_by_model)
        audio_span = _audio_span_from_ctc_spans(
            matched_spans,
            ensemble_results,
            len(audio_array),
            sr,
        )

        # ── Ensemble combine ──────────────────────────────────────────────────
        decision = combine_signals(whisper_result, wav2vec2_result, llr_result, expected_haraka)
        matched_transcript_word = _representative_transcript_word(
            wav2vec2_result.get("transcript_words", [])
        )

        iraab_findings.append({
            "word_index": word_idx,
            "word": focus_word,
            "expected_ending": expected_haraka,
            "detected_ending": decision["detected_ending"],
            "status": decision["status"],
            "detection_confidence": decision["detection_confidence"],
            "audio_span": audio_span,
            "explanation": _build_explanation(
                focus_word, expected_haraka, decision, feedback_language
            ),
            "pronunciation_notes": _build_pronunciation_notes(
                focus_word,
                matched_transcript_word,
                expected_haraka,
                decision,
                feedback_language,
            ),
            "grammar_citations": [],   # populated when reference catalog integrated
            "_debug": {
                "is_focus_word": target["is_focus"],
                "transcript_word": matched_transcript_word,
                "whisper": whisper_result,
                "wav2vec2": wav2vec2_result,
                "llr": {k: round(v, 3) for k, v in llr_result.items()},
                "ctc_word_spans": matched_spans,
                "decision_reason": decision.get("reason"),
                "vote_breakdown": decision.get("vote_breakdown", {}),
            },
        })

    # ── Delivery metrics ──────────────────────────────────────────────────────
    vad_model, vad_utils = ModelLoader.get_silero_vad()
    delivery = compute_delivery_metrics(
        audio=audio_array,
        sr=sr,
        transcript_words=transcript_words,
        vad_model=vad_model,
        vad_utils=vad_utils,
    )

    # ── Build response (Dr's layer2_delivery schema) ──────────────────────────
    response = {
        "reference_text": reference_sentence,
        "iraab_findings": iraab_findings,
        "delivery": delivery,
        "model_provenance": {
            "diacritizer": "reference_diacritic_extraction",
            "aligner": "wav2vec2 CTC argmax word spans (MFA optional)",
            "preprocessing": f"trim_silence + peak_normalize (original {original_duration_ms} ms)",
            "acoustic": "wav2vec2-large-xlsr-53 × 3 + whisper-large-v3 (hypothesis)",
        },
        "exercise_id": exercise_id,
        "transcript": combined_transcript,
    }

    return JSONResponse(content=_json_safe(response))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidate_harakaat(expected: str) -> list[str]:
    """Return the 3 harakaat to test for hypothesis scoring."""
    tanween_group = {"tanwin_fath", "tanwin_damm", "tanwin_kasr"}
    if expected in tanween_group:
        return ["tanwin_fath", "tanwin_damm", "tanwin_kasr", "none"]
    return ["fatha", "damma", "kasra"]


def _json_safe(value):
    """Convert numpy/debug values into plain JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        plain = value.item()
        if isinstance(plain, float) and not np.isfinite(plain):
            return None
        return plain
    return value


def _audio_span_from_ctc_spans(
    spans: list[dict],
    ensemble_results: list[tuple[Optional[str], Optional[np.ndarray]]],
    audio_len: int,
    sr: int,
) -> dict:
    if not spans:
        return {"start_ms": None, "end_ms": None}

    starts = []
    ends = []
    duration_ms = audio_len / sr * 1000.0 if sr else 0.0
    for span in spans:
        frame_count = span.get("frame_count")
        if not frame_count:
            continue
        ms_per_frame = duration_ms / frame_count
        starts.append(span["start_frame"] * ms_per_frame)
        ends.append((span["end_frame"] + 1) * ms_per_frame)

    if not starts or not ends:
        return {"start_ms": None, "end_ms": None}
    return {
        "start_ms": int(max(0, min(starts))),
        "end_ms": int(min(duration_ms, max(ends))),
    }


def _clean_reference_word(word: str) -> str:
    return re.sub(r"^[^\u0600-\u06FF]+|[^\u0600-\u06FF\u064b-\u0652]+$", "", word)


def _build_explanation(
    word: str,
    expected: str,
    decision: dict,
    lang: str,
) -> Optional[str]:
    status    = decision["status"]
    detected  = decision["detected_ending"]
    conf      = decision["detection_confidence"]

    if status == "correct":
        return None   # no feedback needed for correct pronunciation

    ending_ar = {
        "fatha":       "الفتحة (a)",
        "damma":       "الضمة (u)",
        "kasra":       "الكسرة (i)",
        "tanwin_fath": "تنوين الفتح (an)",
        "tanwin_damm": "تنوين الضم (un)",
        "tanwin_kasr": "تنوين الكسر (in)",
    }

    if status == "uncertain":
        if lang == "ar":
            return f"لم يتمكن النظام من التحقق من نطق نهاية الكلمة '{word}' بثقة كافية (ثقة {conf:.0%}). يُرجى الاستماع للتسجيل."
        return f"The system could not verify the ending of '{word}' with sufficient confidence ({conf:.0%}). Please listen to the recording."

    if decision.get("reason") == "implicit_short_vowel_base_match":
        if lang == "ar":
            return f"الكلمة '{word}' تطابق النص المرجعي في التفريغ، والحركة النهائية قصيرة فلم يكتبها نموذج ASR كحرف مستقل. اعتُبرت صحيحة بثقة متوسطة."
        return f"'{word}' matches the reference transcript; the final short vowel was not written as a separate ASR letter, so it is accepted with medium confidence."

    if decision.get("reason") == "expected_aware_tanween" and status == "correct":
        if lang == "ar":
            return None
        return None

    exp_label  = ending_ar.get(expected, expected)
    det_label  = ending_ar.get(detected, detected)

    if lang == "ar":
        return f"الكلمة '{word}' تحتاج {exp_label} لكن يبدو أنك نطقت {det_label}. راجع الإعراب."
    return f"'{word}' requires {exp_label} but {det_label} was detected. Please review the i'rab."


def _representative_transcript_word(words: list[str]) -> Optional[str]:
    if not words:
        return None
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts, key=counts.get)


def _build_pronunciation_notes(
    word: str,
    transcript_word: Optional[str],
    expected: str,
    decision: dict,
    lang: str,
) -> list[str]:
    notes = []

    if decision.get("reason") == "implicit_short_vowel_base_match":
        notes.append(
            "الحركة النهائية قصيرة؛ نماذج ASR العربية غالبًا لا تكتب الضمة/الكسرة/الفتحة كحرف مستقل."
            if lang == "ar"
            else "Final short vowels are often not written by Arabic ASR."
        )

    if expected.startswith("tanwin") and decision["status"] == "incorrect":
        notes.append(
            "تحقق من نطق التنوين بوضوح، خصوصًا صوت النون النهائي."
            if lang == "ar"
            else "Check that tanween is pronounced clearly, especially the final /n/ sound."
        )

    if decision.get("reason") == "expected_aware_tanween":
        notes.append(
            "اعتُبر شكل التفريغ الصوتي مطابقًا للتنوين المتوقع، حتى لو كتبه ASR كحرف مدّ عادي."
            if lang == "ar"
            else "The ASR spelling was interpreted as the expected tanween sound."
        )

    if "\u0651" in word:
        notes.append(
            "هذه الكلمة تحتوي على شدة؛ يجب نطق الحرف مشددًا لا خفيفًا."
            if lang == "ar"
            else "This word contains shadda; the consonant should be geminated."
        )

    if "\u0652" in word:
        notes.append(
            "هذه الكلمة تحتوي على سكون داخلي؛ تجنب إضافة حركة قصيرة بعد الحرف الساكن."
            if lang == "ar"
            else "This word contains sukun; avoid adding a short vowel after the silent consonant."
        )

    if transcript_word:
        notes.append(
            f"الكلمة كما سمعها النظام: {transcript_word}"
            if lang == "ar"
            else f"ASR heard: {transcript_word}"
        )

    return notes
