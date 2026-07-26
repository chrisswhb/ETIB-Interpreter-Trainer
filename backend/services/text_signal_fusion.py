from __future__ import annotations

from difflib import SequenceMatcher
import re

from utils.arabizi import extract_ending_from_arabic, normalize_arabic_word


def fuse_text_signals(response: dict, speech_transcript: dict) -> dict:
    """Attach ASR/Gemma word evidence to acoustic ending findings.

    The acoustic classifier remains the primary pronunciation signal. Text
    signals are used only to clarify low-confidence cases and expose what the
    system actually transcribed from free speech.
    """
    if not response or not speech_transcript:
        return response

    findings = response.get("iraab_findings") or []
    reference_words = _words(response.get("reference_text") or "")
    asr_words = _words(speech_transcript.get("transcript") or "")
    diac_words = _words(speech_transcript.get("diacritized_transcript") or "")
    if not findings or not reference_words:
        return response

    asr_alignment = _align_words(reference_words, asr_words)
    diac_alignment = _align_words(reference_words, diac_words)

    for finding in findings:
        idx = int(finding.get("word_index") or 0)
        expected = finding.get("expected_ending")
        confidence = float(finding.get("detection_confidence") or 0.0)

        asr = asr_alignment.get(idx)
        diac = diac_alignment.get(idx)
        text_ending = extract_ending_from_arabic(diac["word"]) if diac else "none"

        finding["asr_word"] = asr["word"] if asr else None
        finding["asr_match_score"] = round(float(asr["score"]), 3) if asr else 0.0
        finding["diacritized_transcript_word"] = diac["word"] if diac else None
        finding["text_detected_ending"] = text_ending
        finding.setdefault("_debug", {})["text_fusion"] = {
            "asr": asr,
            "diacritized": diac,
            "text_detected_ending": text_ending,
            "rule": "acoustic_primary",
        }

        notes = finding.setdefault("pronunciation_notes", [])
        if asr and asr["score"] < 0.55:
            notes.append("ASR word match is weak; the spoken word may not match the reference.")
            if confidence < 0.65:
                finding["status"] = "uncertain"
                finding.setdefault("_debug", {})["text_fusion"]["rule"] = "weak_asr_word_match"
            continue

        # When acoustic evidence is very weak, use the text layer as a fallback
        # hypothesis. This improves free-speech UX without overriding strong
        # acoustic disagreement.
        if confidence < 0.48 and text_ending != "none":
            finding["detected_ending"] = text_ending
            finding["detection_confidence"] = max(confidence, 0.56)
            finding["status"] = "correct" if text_ending == expected else "incorrect"
            finding["explanation"] = _text_fallback_explanation(
                finding.get("word") or "",
                expected,
                text_ending,
                finding["status"],
            )
            finding.setdefault("_debug", {})["text_fusion"]["rule"] = "low_confidence_text_fallback"
        elif confidence < 0.62 and text_ending == expected:
            notes.append("The ASR+diacritization text layer supports the expected final ending.")
            finding.setdefault("_debug", {})["text_fusion"]["rule"] = "text_supports_expected"
        elif confidence < 0.62 and text_ending not in {"none", expected}:
            notes.append("The ASR+diacritization text layer suggests a different final ending.")
            finding.setdefault("_debug", {})["text_fusion"]["rule"] = "text_disagrees_with_acoustic_or_reference"

    response.setdefault("model_provenance", {})["text_signal_fusion"] = (
        "ASR transcript + diacritized transcript aligned to reference; acoustic remains primary"
    )
    return response


def _text_fallback_explanation(word: str, expected: str, detected: str, status: str) -> str:
    if status == "correct":
        return (
            f"Acoustic confidence for '{word}' was low, so the system used the "
            "ASR+diacritization text layer as a fallback; it supports the expected ending."
        )
    return (
        f"Acoustic confidence for '{word}' was low, and the ASR+diacritization "
        f"text layer suggested {detected} instead of {expected}."
    )


def _words(text: str) -> list[str]:
    return [w for w in (_clean_word(part) for part in text.split()) if w]


def _clean_word(word: str) -> str:
    return re.sub(r"^[^\u0600-\u06FF\u064b-\u0652]+|[^\u0600-\u06FF\u064b-\u0652]+$", "", word or "")


def _align_words(reference_words: list[str], candidate_words: list[str]) -> dict[int, dict]:
    alignment: dict[int, dict] = {}
    used: set[int] = set()
    for i, ref_word in enumerate(reference_words):
        best = None
        for j in range(max(0, i - 3), min(len(candidate_words), i + 4)):
            if j in used:
                continue
            score = _word_score(ref_word, candidate_words[j])
            if best is None or score > best["score"]:
                best = {"word": candidate_words[j], "candidate_index": j, "score": score}
        if best and best["score"] >= 0.35:
            used.add(best["candidate_index"])
            alignment[i] = best
    return alignment


def _word_score(a: str, b: str) -> float:
    na = normalize_arabic_word(a)
    nb = normalize_arabic_word(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()
