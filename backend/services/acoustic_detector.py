"""
acoustic_detector.py — Layer 2 acoustic i'rab detection.

Implements your 3-signal ensemble:
  Signal 1: Competing Whisper hypothesis acoustic scoring (primary — reads logprob, NOT text)
  Signal 2: wav2vec2 CTC ending-letter detection (3-model vote)
  Signal 3: Log-Likelihood Ratio (LLR) logit analysis on word-boundary frames

+ Dr's additions:
  - CATT/CAMeL diacritizer for expected_ending ground truth
  - Confidence-gated status (correct | incorrect | uncertain)
  - detection_confidence float 0-1 for the schema
"""

import os
import re
import math
import logging
from difflib import SequenceMatcher
import numpy as np
from typing import Optional

from utils.arabizi import (
    arabic_to_arabizi, extract_ending, arabizi_to_haraka_label,
    haraka_to_arabizi, extract_ending_from_arabic, HARAKA_LABELS,
    normalize_arabic_word, remove_final_long_vowel_marker,
)

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Confidence thresholds (Dr's guidance: damma/tanwin_damm near 20-30% → uncertain)
UNCERTAIN_THRESHOLD = 0.45
DAMMA_UNCERTAIN_THRESHOLD = 0.55  # higher bar for notoriously hard damma


# ── Haraka variants for hypothesis scoring ────────────────────────────────────

HARAKA_ENDINGS = {
    "none":         "",
    "fatha":       "\u064e",   # َ
    "damma":       "\u064f",   # ُ
    "kasra":       "\u0650",   # ِ
    "tanwin_fath": "\u064b",   # ً
    "tanwin_damm": "\u064c",   # ٌ
    "tanwin_kasr": "\u064d",   # ٍ
}

HARAKA_ARABIZI = {
    "fatha": "a", "damma": "u", "kasra": "i",
    "tanwin_fath": "an", "tanwin_damm": "un", "tanwin_kasr": "in",
}


# ── Signal 1: Competing Whisper Hypothesis Scoring ────────────────────────────

def build_hypothesis_sentence(reference_sentence: str, focus_word: str, target_haraka: str) -> str:
    """
    Replace the focus word's final haraka with target_haraka.
    Used to generate the 3 competing hypotheses (a/u/i or an/un/in).
    """
    ending_char = HARAKA_ENDINGS.get(target_haraka, "")
    # Remove last diacritic from focus word and append the target haraka
    base_word = re.sub(r"[\u064b-\u0652]+$", "", focus_word)
    new_word  = base_word + ending_char

    return reference_sentence.replace(focus_word, new_word, 1)


async def score_hypothesis_whisper(audio_bytes: bytes, hypothesis_text: str) -> float:
    """
    Score how well audio acoustically matches hypothesis_text using Groq Whisper.
    Returns avg_logprob (higher = better match). We NEVER use the text output.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — Whisper hypothesis scoring disabled")
        return -float("inf")

    try:
        import httpx, tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                with open(tmp_path, "rb") as f:
                    resp = await client.post(
                        f"{GROQ_BASE_URL}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        data={
                            "model": "whisper-large-v3",
                            "response_format": "verbose_json",
                            "prompt": hypothesis_text,   # acoustic hint — not grammar correction
                            "language": "ar",
                        },
                        files={"file": ("audio.wav", f, "audio/wav")},
                    )
            if resp.status_code != 200:
                logger.warning(f"Groq API error {resp.status_code}: {resp.text[:200]}")
                return -float("inf")

            data = resp.json()
            # Extract avg_logprob from the verbose response
            segments = data.get("segments", [])
            if not segments:
                return data.get("avg_logprob", -float("inf"))

            total_logprob = sum(s.get("avg_logprob", 0) for s in segments)
            return total_logprob / len(segments)

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        logger.warning(f"Whisper hypothesis scoring failed: {e}")
        return -float("inf")


def whisper_score_to_weight(best_logprob: float, second_logprob: float) -> float:
    """
    Scale logprob margin to a vote weight in [3, 8].
    Larger margin → more confidence → higher weight.
    """
    margin = abs(best_logprob - second_logprob)
    return min(3.0 + margin * 10, 8.0)


# ── Signal 2: wav2vec2 Ending-Letter Detection ────────────────────────────────

def detect_ending_from_wav2vec2_text(transcription: str, focus_arabizi: str) -> Optional[str]:
    """
    Check the last letter written by wav2vec2 for the focus word to infer the haraka.
    Alef ا → fatha, Waw و → damma, Ya ي → kasra.
    """
    # Normalise and find focus word region in transcription
    text = transcription.strip()
    return detect_ending_from_transcribed_word(text)


def detect_ending_from_transcribed_word(word: str) -> Optional[str]:
    """
    Infer a pronounced ending from one wav2vec2 transcribed word.

    This intentionally uses the acoustic transcript spelling, not the reference grammar:
    المديرة / الاجتماعة -> fatha, ذهبو -> damma, مديري -> kasra.
    """
    text = word.strip()
    if text.endswith("ان"):
        return "tanwin_fath"
    if text.endswith("ون"):
        return "tanwin_damm"
    if text.endswith("ين"):
        return "tanwin_kasr"
    if text.endswith("ن"):
        if text.endswith(("ان", "اً")):
            return "tanwin_fath"
        if text.endswith(("ون", "ٌ")):
            return "tanwin_damm"
        if text.endswith(("ين", "ٍ")):
            return "tanwin_kasr"

    for arabic_letter, haraka in [
        ("\u0627", "fatha"),   # alef
        ("\u0649", "fatha"),   # alef maqsura
        ("\u0629", "fatha"),   # ta marbuta often appears when final /a/ is heard
        ("\u0648", "damma"),   # waw
        ("\u064a", "kasra"),   # ya
    ]:
        if text.endswith(arabic_letter):
            return haraka
    return None


def refine_detected_ending_for_expected(
    transcript_word: str,
    detected: Optional[str],
    expected_haraka: str | None,
) -> tuple[Optional[str], str | None]:
    """
    ASR often writes tanween as ordinary final letters:
      طالبا  for طالبًا, طالبين for طالبٍ, طالبون for طالبٌ.
    If the expected category is tanween, reinterpret those spellings as the
    intended tanween sound instead of a plain short-vowel ending.
    """
    if not expected_haraka:
        return detected, None

    text = normalize_arabic_word(transcript_word)
    if expected_haraka == "tanwin_fath" and text.endswith(("ا", "ان")):
        return "tanwin_fath", "expected_aware_tanween_fath"
    if expected_haraka == "tanwin_kasr" and text.endswith(("ي", "ين")):
        return "tanwin_kasr", "expected_aware_tanween_kasr"
    if expected_haraka == "tanwin_damm" and text.endswith(("و", "ون")):
        return "tanwin_damm", "expected_aware_tanween_damm"

    return detected, None


def wav2vec2_ensemble_vote(
    ensemble_outputs: list[tuple[str, np.ndarray | None]],
    focus_word_arabizi: str,
    word_index: int | None = None,
    reference_word: str | None = None,
    expected_haraka: str | None = None,
) -> dict:
    """
    Run 3-model ensemble vote for ending detection.
    Returns: { haraka: str, vote_count: int, total_weight: float }
    """
    votes: dict[str, float] = {}
    implicit_matches = 0
    transcript_words = []
    refinements = []

    for model_text, logits in ensemble_outputs:
        if model_text is None:
            continue
        words = model_text.split()
        transcript_word = _select_transcript_word(reference_word, words, word_index)
        if transcript_word:
            transcript_words.append(transcript_word)
            detected = detect_ending_from_transcribed_word(transcript_word)
            detected, refinement = refine_detected_ending_for_expected(
                transcript_word,
                detected,
                expected_haraka,
            )
            if refinement:
                refinements.append(refinement)
            if (
                detected is None
                and reference_word
                and expected_haraka
                and _same_word_without_written_ending(reference_word, transcript_word)
            ):
                implicit_matches += 1
        else:
            detected = detect_ending_from_wav2vec2_text(model_text, focus_word_arabizi)
            detected, refinement = refine_detected_ending_for_expected(
                model_text,
                detected,
                expected_haraka,
            )
            if refinement:
                refinements.append(refinement)
        if detected:
            votes[detected] = votes.get(detected, 0) + 3.0  # 3 pts per model

    if not votes:
        if implicit_matches and expected_haraka:
            tanween = {"tanwin_fath", "tanwin_damm", "tanwin_kasr"}
            if expected_haraka in tanween:
                return {
                    "haraka": "none",
                    "vote_count": implicit_matches,
                    "total_weight": implicit_matches * 2.0,
                    "source": "tanween_missing_in_transcript",
                    "refinements": refinements,
                    "transcript_words": transcript_words,
                }
            return {
                "haraka": expected_haraka,
                "vote_count": implicit_matches,
                "total_weight": implicit_matches * 2.0,
                "source": "implicit_short_vowel_base_match",
                "refinements": refinements,
                "transcript_words": transcript_words,
            }
        return {"haraka": None, "vote_count": 0, "total_weight": 0.0}

    best = max(votes, key=votes.get)
    source = "expected_aware_tanween" if refinements else "explicit_written_ending"
    return {
        "haraka": best,
        "vote_count": int(votes[best] / 3.0),
        "total_weight": votes[best],
        "source": source,
        "refinements": refinements,
        "transcript_words": transcript_words,
    }


def _same_word_without_written_ending(reference_word: str, transcript_word: str) -> bool:
    ref = normalize_arabic_word(reference_word)
    hyp = normalize_arabic_word(transcript_word)
    hyp_without_ending = remove_final_long_vowel_marker(transcript_word)
    return bool(ref and hyp and (hyp == ref or hyp_without_ending == ref))


def _select_transcript_word(
    reference_word: str | None,
    words: list[str],
    preferred_index: int | None,
) -> Optional[str]:
    if not words:
        return None

    if preferred_index is not None and 0 <= preferred_index < len(words):
        exact = words[preferred_index]
        if not reference_word:
            return exact
        if _is_plausible_match(reference_word, exact):
            return exact

    if not reference_word:
        return None

    best_word = None
    best_score = 0.0
    for idx, word in enumerate(words):
        score = _word_match_score(reference_word, word)
        if preferred_index is not None:
            distance = abs(idx - preferred_index)
            score -= min(distance, 3) * 0.04
        if score > best_score:
            best_word = word
            best_score = score

    return best_word if best_score >= 0.72 else None


def _is_plausible_match(reference_word: str, transcript_word: str) -> bool:
    return _word_match_score(reference_word, transcript_word) >= 0.72


def _word_match_score(reference_word: str, transcript_word: str) -> float:
    ref = normalize_arabic_word(reference_word)
    hyp = normalize_arabic_word(transcript_word)
    hyp_base = remove_final_long_vowel_marker(transcript_word)
    if not ref or not hyp:
        return 0.0
    if hyp == ref or hyp_base == ref:
        return 1.0
    if ref.startswith("ال") and hyp == ref[2:]:
        return 0.92
    if hyp.startswith("ال") and hyp[2:] == ref:
        return 0.92
    return max(
        SequenceMatcher(None, ref, hyp).ratio(),
        SequenceMatcher(None, ref, hyp_base).ratio(),
    )


def ctc_word_spans_from_logits(
    logits: np.ndarray,
    vocab: list[str],
    blank_tokens: set[str] | None = None,
) -> list[dict]:
    """
    Build approximate word spans from CTC argmax frames.

    This is a lightweight forced-alignment substitute: it collapses repeated
    CTC tokens, splits on word delimiters, and keeps the first/last frame for
    each decoded word. It is not as strong as MFA, but it is much better than
    splitting frames evenly across the sentence.
    """
    if logits is None or len(logits) == 0:
        return []

    blank_tokens = blank_tokens or {"<pad>", "[PAD]", "<s>", "</s>", ""}
    delimiter_tokens = {"|", " ", "▁"}
    pred_ids = np.argmax(logits, axis=-1)

    spans: list[dict] = []
    chars: list[str] = []
    start_frame: int | None = None
    end_frame: int | None = None
    prev_id: int | None = None

    def close_word() -> None:
        nonlocal chars, start_frame, end_frame
        word = "".join(chars).strip()
        if word and start_frame is not None and end_frame is not None:
            spans.append({
                "word": word,
                "normalized": normalize_arabic_word(word),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            })
        chars = []
        start_frame = None
        end_frame = None

    for frame_idx, token_id in enumerate(pred_ids):
        token_id = int(token_id)
        if token_id == prev_id:
            continue
        prev_id = token_id
        token = vocab[token_id] if 0 <= token_id < len(vocab) else ""
        token = token or ""
        if token in blank_tokens or token.startswith("<"):
            continue
        if token in delimiter_tokens:
            close_word()
            continue
        if token.startswith("##"):
            token = token[2:]
        if token.startswith("▁"):
            close_word()
            token = token[1:]
        if not token:
            continue
        if start_frame is None:
            start_frame = frame_idx
        end_frame = frame_idx
        chars.append(token)

    close_word()
    return spans


def select_ctc_word_span(
    reference_word: str,
    spans: list[dict],
    preferred_index: int | None = None,
) -> Optional[dict]:
    """Pick the CTC word span that best matches the reference word."""
    best_span = None
    best_score = 0.0
    for idx, span in enumerate(spans):
        score = _word_match_score(reference_word, span.get("word", ""))
        if preferred_index is not None:
            score -= min(abs(idx - preferred_index), 4) * 0.035
        if score > best_score:
            best_score = score
            best_span = span
    if best_span and best_score >= 0.68:
        result = dict(best_span)
        result["match_score"] = round(best_score, 3)
        return result
    return None


# ── Signal 3: LLR Logit Analysis ─────────────────────────────────────────────

def compute_llr_from_logits(
    logits: np.ndarray,
    vocab: list[str],
    n_frames: int = 8,
    word_index: int | None = None,
    total_words: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> dict[str, float]:
    """
    Compute Log-Likelihood Ratio for ending characters on the last n_frames frames.

    LLR(char) = log P(char|audio) - log P_prior(char)
    P_prior correction removes frequency bias (alef >> waw in Arabic text).
    """
    # Character prior frequencies (estimated from Arabic text corpora)
    PRIORS = {
        "\u0627": 0.15,   # alef — very common
        ("\u0649"): 0.03,  # alef maqsura
        "\u0648": 0.06,   # waw
        "\u064a": 0.08,   # ya
    }

    ending_chars = {
        "\u0627": "fatha",
        "\u0649": "fatha",
        "\u0648": "damma",
        "\u064a": "kasra",
    }

    if logits is None or len(logits) == 0:
        return {}

    if frame_start is not None and frame_end is not None:
        start = max(0, min(int(frame_start), len(logits) - 1))
        end = max(start + 1, min(int(frame_end) + 1, len(logits)))
        segment = logits[start:end]
        frames = segment[-min(n_frames, len(segment)):]
    elif word_index is not None and total_words and total_words > 0:
        start = int(len(logits) * word_index / total_words)
        end = int(len(logits) * (word_index + 1) / total_words)
        end = max(start + 1, min(end, len(logits)))
        segment = logits[start:end]
        frames = segment[-min(n_frames, len(segment)):]
    else:
        frames = logits[-n_frames:]  # shape: (n_frames, vocab_size)
    log_probs = frames - frames.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    avg_log_prob = log_probs.mean(axis=0)

    llr_scores: dict[str, float] = {}
    for ch, haraka in ending_chars.items():
        if ch in vocab:
            idx = vocab.index(ch)
            prior = PRIORS.get(ch, 0.05)
            llr = float(avg_log_prob[idx]) - math.log(prior)
            # Accumulate per haraka
            llr_scores[haraka] = llr_scores.get(haraka, 0) + llr

    return llr_scores


def merge_llr_scores(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average LLR scores from all available wav2vec2 models."""
    merged: dict[str, list[float]] = {}
    for scores in score_dicts:
        for haraka, score in scores.items():
            merged.setdefault(haraka, []).append(score)
    return {haraka: float(np.mean(values)) for haraka, values in merged.items()}


# ── Expected Ending (Dr's Layer 2 Step 1) ────────────────────────────────────

def get_expected_ending_from_reference(focus_word: str) -> str:
    """
    Extract the grammatically expected ending from the diacritized reference word.
    Uses the diacritic already present on the word (reliable since reference is pre-diacritized).

    Dr's spec: use CATT / CAMeL Tools for live text; here we read the diacritic directly
    since the reference JSON already has fully vocalised Arabic.
    """
    return extract_ending_from_arabic(focus_word)


# ── Ensemble Combiner ─────────────────────────────────────────────────────────

def combine_signals(
    whisper_result: dict,   # {haraka, weight}
    wav2vec2_result: dict,  # {haraka, vote_count, total_weight}
    llr_result: dict,       # {haraka: llr_score}
    expected_haraka: str,
) -> dict:
    """
    Combine 3 signals into a final decision with confidence-gated status.

    Signal weights (from your technical report):
      Whisper hypothesis : 3–8 (scales with logprob margin)
      wav2vec2 write     : 3 per model (up to 9)
      LLR logit          : 0.4
    """
    votes: dict[str, float] = {}

    # Signal 1 — Whisper hypothesis (primary)
    if whisper_result.get("haraka"):
        h = whisper_result["haraka"]
        votes[h] = votes.get(h, 0) + whisper_result.get("weight", 3.0)

    # Signal 2 — wav2vec2 write
    if wav2vec2_result.get("haraka"):
        h = wav2vec2_result["haraka"]
        votes[h] = votes.get(h, 0) + wav2vec2_result.get("total_weight", 0)

    # Signal 3 — relative LLR. Scores are often all negative; the least negative
    # ending is still useful if it clearly beats the alternatives.
    if llr_result:
        sorted_llr = sorted(llr_result.items(), key=lambda x: x[1], reverse=True)
        best_llr_h, best_llr_s = sorted_llr[0]
        second_llr_s = sorted_llr[1][1] if len(sorted_llr) > 1 else best_llr_s - 1.0
        margin = best_llr_s - second_llr_s
        if margin > 0.08:
            votes[best_llr_h] = votes.get(best_llr_h, 0) + min(3.0, 0.8 + margin * 1.8)

    if not votes:
        return {
            "detected_ending": "none",
            "status": "uncertain",
            "detection_confidence": 0.0,
        }

    total = sum(votes.values())
    best = max(votes, key=votes.get)
    confidence = votes[best] / total if total > 0 else 0.0
    if wav2vec2_result.get("source") == "implicit_short_vowel_base_match":
        implicit_cap = min(0.84, 0.72 + 0.04 * wav2vec2_result.get("vote_count", 1))
        confidence = min(confidence, implicit_cap) if best == expected_haraka else confidence
    elif wav2vec2_result.get("source") == "expected_aware_tanween":
        confidence = min(confidence, 0.88)
    elif wav2vec2_result.get("source") == "tanween_missing_in_transcript":
        confidence = min(confidence, 0.7)

    # Dr's confidence-gating: damma/tanwin_damm are hard — report as uncertain if below threshold
    threshold = DAMMA_UNCERTAIN_THRESHOLD if best in ("damma", "tanwin_damm") else UNCERTAIN_THRESHOLD
    if confidence < threshold:
        status = "uncertain"
    elif best == expected_haraka:
        status = "correct"
    else:
        status = "incorrect"

    return {
        "detected_ending": best,
        "status": status,
        "detection_confidence": round(confidence, 3),
        "reason": wav2vec2_result.get("source"),
        "vote_breakdown": {k: round(v, 2) for k, v in votes.items()},
    }
