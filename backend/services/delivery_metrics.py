"""
delivery_metrics.py — Layer 2 delivery analysis using Silero VAD.

Dr's Layer 2 spec:
  - total_duration_ms
  - speech_rate_wpm (approx)
  - pauses: list of {start_ms, end_ms, duration_ms}
  - long_pause_count (pauses > 500ms)
  - filled_pauses (filler tokens from wav2vec2 transcript)
  - repetitions (n-gram repeats in transcript)
  - false_starts (restart patterns in transcript)

Silero VAD is the free/open choice recommended by Dr.
"""

import re
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

LONG_PAUSE_THRESHOLD_MS = 500
SAMPLE_RATE = 16000

FILLER_TOKENS_AR = {"آه", "أم", "إم", "هم", "مم", "أمم", "إمم", "يعني", "عم", "هاه"}
FILLER_TOKENS_EN = {"um", "uh", "er", "ah", "hmm", "like", "you know"}


def compute_delivery_metrics(
    audio: np.ndarray,
    sr: int,
    transcript_words: list[str],
    vad_model=None,
    vad_utils=None,
    duration_ms: Optional[int] = None,
) -> dict:
    """
    Compute full delivery metrics as specified in Dr's Layer 2 JSON schema.

    Returns a dict matching the 'delivery' object in layer2_delivery schema.
    """
    if duration_ms is None:
        duration_ms = int(len(audio) / sr * 1000)

    pauses    = _detect_pauses(audio, sr, vad_model, vad_utils)
    long_pauses = [p for p in pauses if p["duration_ms"] >= LONG_PAUSE_THRESHOLD_MS]

    word_count = len([w for w in transcript_words if w.strip()])
    duration_s = duration_ms / 1000
    speech_rate_wpm = round((word_count / duration_s) * 60, 1) if duration_s > 0 else 0.0

    filled = _detect_filled_pauses(transcript_words)
    repetitions = _detect_repetitions(transcript_words)
    false_starts = _detect_false_starts(transcript_words)

    return {
        "duration_ms": duration_ms,
        "speech_rate_wpm": speech_rate_wpm,
        "pauses": pauses,
        "long_pause_count": len(long_pauses),
        "filled_pauses": filled,
        "repetitions": repetitions,
        "false_starts": false_starts,
        "number_reading_errors": [],  # populated separately by caller if numeral comparison needed
    }


def _detect_pauses(audio: np.ndarray, sr: int, vad_model=None, vad_utils=None) -> list[dict]:
    """Use Silero VAD to find silent gaps between speech segments."""
    if vad_model is None:
        return _fallback_energy_pauses(audio, sr)

    try:
        import torch
        (get_speech_timestamps, _, _, _, _) = vad_utils

        tensor = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sr)

        pauses = []
        prev_end = 0
        for seg in timestamps:
            start_ms = int(seg["start"] / sr * 1000)
            if start_ms - prev_end > 100:  # ignore tiny gaps < 100ms
                pauses.append({
                    "start_ms": prev_end,
                    "end_ms": start_ms,
                    "duration_ms": start_ms - prev_end,
                })
            prev_end = int(seg["end"] / sr * 1000)

        return pauses

    except Exception as e:
        logger.warning(f"Silero VAD detection failed, using energy fallback: {e}")
        return _fallback_energy_pauses(audio, sr)


def _fallback_energy_pauses(audio: np.ndarray, sr: int) -> list[dict]:
    """Energy-based pause detection (fallback when VAD unavailable)."""
    frame_ms   = 20
    frame_size = int(sr * frame_ms / 1000)
    threshold  = 0.01

    pauses = []
    in_pause = False
    pause_start = 0

    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        t_ms = int(i / sr * 1000)

        if energy < threshold and not in_pause:
            in_pause = True
            pause_start = t_ms
        elif energy >= threshold and in_pause:
            duration = t_ms - pause_start
            if duration > 150:
                pauses.append({"start_ms": pause_start, "end_ms": t_ms, "duration_ms": duration})
            in_pause = False

    return pauses


def _detect_filled_pauses(words: list[str]) -> list[dict]:
    """Find filler tokens in transcript."""
    filled = []
    all_fillers = FILLER_TOKENS_AR | FILLER_TOKENS_EN
    for i, w in enumerate(words):
        if w.strip().lower() in all_fillers:
            filled.append({"time_ms": None, "token": w})  # time_ms set by alignment if available
    return filled


def _detect_repetitions(words: list[str], n: int = 2) -> list[dict]:
    """Detect n-gram repetitions in word sequence."""
    reps = []
    for size in range(1, n + 1):
        ngrams = [" ".join(words[i:i+size]) for i in range(len(words) - size)]
        seen = set()
        for i, gram in enumerate(ngrams):
            if gram in seen:
                reps.append({"span": gram, "time_ms": None})
            seen.add(gram)
    return reps


def _detect_false_starts(words: list[str]) -> list[dict]:
    """
    Detect restart patterns: word followed by self-correction marker or immediate repetition.
    Simple heuristic: if same word appears within 2-word window.
    """
    false_starts = []
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 2:
            false_starts.append({"span": words[i], "time_ms": None})
    return false_starts
