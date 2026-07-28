import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from services.acoustic_detector import (
    build_hypothesis_sentence,
    ctc_word_spans_from_logits,
    get_expected_ending_from_reference,
    score_hypothesis_whisper,
)
from utils.audio_utils import normalize_for_asr, trim_silence
from services.ending_acoustic_features import ending_acoustic_features

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = Path(os.environ.get("ETIB_WAV2VEC2_MODEL_DIR", Path.home() / "Desktop" / "etib_models" / "wav2vec2_jonatasgrosman_arabic")).expanduser()
REMOTE_MODEL_ID = os.environ.get("ETIB_WAV2VEC2_MODEL_ID", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
CLASSIFIER_PATH = (
    ROOT
    / "arabic_sentence_ending_training"
    / "models"
    / "sentence_ending_classifier.joblib"
)
AUGMENTED_CLASSIFIER_PATH = (
    ROOT
    / "arabic_sentence_ending_training"
    / "models"
    / "sentence_ending_classifier_acoustic_augmented.joblib"
)
BASE_VOWEL_CLASSIFIER_PATH = (
    ROOT
    / "arabic_sentence_ending_training"
    / "models"
    / "sentence_base_vowel_classifier.joblib"
)
FALLBACK_CLASSIFIER_PATH = (
    ROOT
    / "arabic_ending_training"
    / "models"
    / "all_plus_boost93_boost85_best_classifier.joblib"
)

ENDING_LABELS_AR = {
    "fatha": "فتحة",
    "damma": "ضمة",
    "kasra": "كسرة",
    "none": "بدون حركة",
    "tanwin_fath": "تنوين فتح",
    "tanwin_damm": "تنوين ضم",
    "tanwin_kasr": "تنوين كسر",
}


class TrainedEndingDetector:
    _processor = None
    _wav2vec2 = None
    _classifier = None
    _base_classifier = None
    _fallback_classifier = None
    _ctc_processor = None
    _ctc_model = None

    @classmethod
    def load(cls) -> None:
        if cls._processor is not None and cls._wav2vec2 is not None and cls._classifier is not None:
            return
        model_source, local_only = _wav2vec2_source()
        classifier_path = _active_classifier_path()
        if not classifier_path.exists():
            raise RuntimeError(f"trained classifier not found: {classifier_path}")

        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        cls._processor = Wav2Vec2Processor.from_pretrained(
            model_source, local_files_only=local_only
        )
        cls._wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_source, local_files_only=local_only
        )
        cls._wav2vec2.eval()
        cls._classifier = joblib.load(classifier_path)
        cls._base_classifier = None
        if not _uses_augmented_classifier() and BASE_VOWEL_CLASSIFIER_PATH.exists():
            cls._base_classifier = joblib.load(BASE_VOWEL_CLASSIFIER_PATH)
        # The isolated-word fallback was trained under a different recording
        # setup and hurts live sentence decisions. Keep the sentence-tail model
        # as the single exact-ending authority.
        cls._fallback_classifier = None
        logger.info("Loaded trained Arabic ending classifier: %s", classifier_path)

    @classmethod
    async def transcribe_word_timestamps(cls, wav_bytes: bytes) -> list[dict]:
        if os.environ.get("ENABLE_GROQ_WORD_TIMESTAMPS") != "1":
            return []
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return []

        tmp_path = None
        try:
            import httpx

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            async with httpx.AsyncClient(timeout=45) as client:
                with open(tmp_path, "rb") as f:
                    resp = await client.post(
                        f"{GROQ_BASE_URL}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        data=[
                            ("model", "whisper-large-v3"),
                            ("response_format", "verbose_json"),
                            ("language", "ar"),
                            ("timestamp_granularities[]", "word"),
                        ],
                        files={"file": ("audio.wav", f, "audio/wav")},
                    )
            if resp.status_code != 200:
                logger.warning("Groq timestamp transcription failed %s: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            words = data.get("words") or []
            spans = []
            for item in words:
                word = str(item.get("word", "")).strip()
                start = item.get("start")
                end = item.get("end")
                if word and start is not None and end is not None:
                    spans.append({"word": word, "start": float(start), "end": float(end), "source": "groq_whisper_word_timestamps"})
            return spans
        except Exception as exc:
            logger.warning("Groq word timestamp alignment unavailable: %s", exc)
            return []
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @classmethod
    def ctc_word_timestamps(cls, audio: np.ndarray, sr: int) -> list[dict]:
        """Approximate forced alignment from local wav2vec2 CTC frames.

        The CTC model does not need to output tashkeel. It only gives rough word
        spans, which are then matched against the known reference sentence.
        """
        cls._load_ctc_aligner()
        if cls._ctc_processor is None or cls._ctc_model is None or len(audio) == 0:
            return []

        try:
            import torch

            inputs = cls._ctc_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                outputs = cls._ctc_model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

            vocab = cls._ctc_processor.tokenizer.get_vocab()
            vocab_list = [""] * (max(vocab.values()) + 1)
            for token, idx in vocab.items():
                if 0 <= idx < len(vocab_list):
                    vocab_list[idx] = token

            spans = ctc_word_spans_from_logits(logits, vocab_list)
            duration_s = len(audio) / sr if sr else 0
            frame_count = max(1, len(logits))
            out = []
            for span in spans:
                start = max(0.0, float(span["start_frame"]) / frame_count * duration_s)
                end = min(duration_s, float(span["end_frame"] + 1) / frame_count * duration_s)
                if end > start:
                    out.append({"word": span.get("word", ""), "start": start, "end": end, "source": "wav2vec2_ctc"})
            return out
        except Exception as exc:
            logger.warning("wav2vec2 CTC alignment unavailable: %s", exc)
            return []

    @classmethod
    def _load_ctc_aligner(cls) -> None:
        if cls._ctc_processor is not None and cls._ctc_model is not None:
            return
        model_source, local_only = _wav2vec2_source()
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            cls._ctc_processor = Wav2Vec2Processor.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
            cls._ctc_model = Wav2Vec2ForCTC.from_pretrained(
                model_source,
                local_files_only=local_only,
            )
            cls._ctc_model.eval()
            logger.info("Loaded wav2vec2 CTC aligner: %s", model_source)
        except Exception as exc:
            logger.warning("Could not load wav2vec2 CTC aligner: %s", exc)
            cls._ctc_processor = None
            cls._ctc_model = None

    @classmethod
    def analyze_sentence(
        cls,
        audio: np.ndarray,
        sr: int,
        reference_sentence: str,
        word_spans: Optional[list[dict]] = None,
    ) -> dict:
        cls.load()
        audio = normalize_for_asr(trim_silence(audio, threshold=0.008))
        words = [_clean_word(w) for w in reference_sentence.split()]
        words = [w for w in words if w]
        targets = [
            {
                "word_index": i,
                "word": word,
                "expected": get_expected_ending_from_reference(word),
            }
            for i, word in enumerate(words)
            if _is_gradable_reference_word(word)
        ]

        aligned_segments = _align_timestamp_spans(words, word_spans or [], len(audio), sr)
        segments = aligned_segments or _segment_by_reference_timing(audio, sr, words)
        segmentation_source = _timestamp_source(word_spans or []) if aligned_segments else "energy/reference-weighted"
        findings = []
        for target in targets:
            idx = target["word_index"]
            start, end = segments[idx] if idx < len(segments) else (0, len(audio))
            segment = audio[start:end]
            if len(segment) < int(sr * 0.08):
                detected = "none"
                confidence = 0.0
                probs = {}
                base_detected = "none"
                base_confidence = 0.0
                base_probs = {}
            else:
                detected, confidence, probs, base_detected, base_confidence, base_probs = cls._predict_segment(segment, sr)
                detected, confidence = _apply_base_vowel_fallback(
                    target["expected"],
                    detected,
                    confidence,
                    base_detected,
                    base_confidence,
                )

            status = "correct" if detected == target["expected"] else "incorrect"
            findings.append({
                "word_index": idx,
                "word": target["word"],
                "expected_ending": target["expected"],
                "detected_ending": detected,
                "status": status,
                "detection_confidence": confidence,
                "audio_span": {
                    "start_ms": int(start / sr * 1000),
                    "end_ms": int(end / sr * 1000),
                },
                "explanation": _explain(target["word"], target["expected"], detected, status),
                "grammar_citations": [],
                "_debug": {
                    "classifier": _active_classifier_path().name,
                    "fallback_classifier": (
                        FALLBACK_CLASSIFIER_PATH.name
                        if cls._fallback_classifier is not None
                        else None
                    ),
                    "base_vowel_classifier": (
                        BASE_VOWEL_CLASSIFIER_PATH.name
                        if cls._base_classifier is not None
                        else None
                    ),
                    "base_vowel_detected": base_detected,
                    "base_vowel_confidence": round(float(base_confidence), 4),
                    "base_vowel_probabilities": base_probs,
                    "top_probabilities": probs,
                    "segment_samples": [int(start), int(end)],
                    "segmentation": segmentation_source,
                    "word_timestamps": word_spans or [],
                },
            })

        return {
            "reference_text": reference_sentence,
            "iraab_findings": findings,
            "delivery": {
                "duration_ms": int(len(audio) / sr * 1000) if sr else 0,
                "speech_rate_wpm": round(len(words) / max(len(audio) / sr / 60, 0.01), 1) if sr else 0,
                "pauses": [],
                "long_pause_count": 0,
                "filled_pauses": [],
                "repetitions": [],
                "false_starts": [],
                "number_reading_errors": [],
            },
            "model_provenance": {
                "acoustic": "wav2vec2_jonatasgrosman_arabic embeddings + trained RBF SVM classifier",
                "classifier": str(_active_classifier_path().name),
                "features": (
                    "wav2vec2 multi-tail embeddings + 5ms acoustic energy/spectral/pitch cues"
                    if _uses_augmented_classifier()
                    else "wav2vec2 multi-tail embeddings"
                ),
                "fallback_classifier": (
                    FALLBACK_CLASSIFIER_PATH.name
                    if cls._fallback_classifier is not None
                    else None
                ),
                "base_vowel_classifier": (
                    BASE_VOWEL_CLASSIFIER_PATH.name
                    if cls._base_classifier is not None
                    else None
                ),
                "segmentation": segmentation_source,
            },
            "transcript": " ".join(strip_final_diacritics(w) for w in words),
        }

    @classmethod
    async def apply_sentence_hypothesis_scoring(cls, response: dict, wav_bytes: bytes) -> dict:
        """Use whole-sentence Whisper acoustic scoring to avoid fragile word cuts."""
        if os.environ.get("ENABLE_GROQ_SENTENCE_SCORING") != "1":
            return response
        if not os.environ.get("GROQ_API_KEY"):
            return response

        reference = response.get("reference_text", "")
        findings = response.get("iraab_findings", [])
        if not reference or not findings:
            return response

        for finding in findings:
            expected = finding.get("expected_ending", "none")
            candidates = _candidate_endings(expected)
            if not candidates:
                continue

            scores = {}
            for candidate in candidates:
                hypothesis = build_hypothesis_sentence(reference, finding["word"], candidate)
                scores[candidate] = await score_hypothesis_whisper(wav_bytes, hypothesis)

            finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
            if not finite_scores:
                continue

            ranked = sorted(finite_scores.items(), key=lambda item: item[1], reverse=True)
            best_label, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else best_score - 0.05
            margin = max(0.0, float(best_score - second_score))

            # Whisper logprob margins are small. Keep confidence moderate unless
            # the whole-sentence hypothesis clearly wins.
            confidence = min(0.96, max(0.52, 0.52 + margin * 8.0))
            local_debug = finding.setdefault("_debug", {})
            local_debug["sentence_hypothesis_scores"] = {
                label: round(float(score), 4) for label, score in finite_scores.items()
            }
            local_debug["sentence_hypothesis_margin"] = round(margin, 4)
            local_debug["local_classifier_detected"] = finding.get("detected_ending")
            local_debug["sentence_hypothesis_best"] = best_label

            current_confidence = float(finding.get("detection_confidence") or 0.0)
            if margin < 0.06 or current_confidence > confidence + 0.08:
                local_debug["decision_source"] = "local_classifier_sentence_score_too_close"
                continue

            local_debug["decision_source"] = "groq_sentence_hypothesis"

            finding["detected_ending"] = best_label
            finding["detection_confidence"] = confidence
            finding["status"] = "correct" if best_label == expected else "incorrect"
            finding["explanation"] = _explain(
                finding["word"],
                expected,
                best_label,
                finding["status"],
            )

        response.setdefault("model_provenance", {})["sentence_scoring"] = (
            "Groq whisper-large-v3 whole-sentence competing hypotheses per word"
        )
        return response

    @classmethod
    def _predict_segment(cls, audio: np.ndarray, sr: int) -> tuple[str, float, dict, str, float, dict]:
        import torch

        audio = normalize_for_asr(trim_silence(audio, threshold=0.006))
        speech_start, speech_end = _speech_bounds(audio, sr)
        if speech_end > speech_start:
            audio = audio[speech_start:speech_end]
        vectors = cls._ending_tail_vectors(audio, sr)
        probabilities = _average_probability_dicts([
            cls._probabilities(cls._classifier, vector) for vector in vectors
        ])

        base_detected = "none"
        base_confidence = 0.0
        base_ranked = {}
        if cls._base_classifier is not None and vectors:
            base_probabilities = _average_probability_dicts([
                cls._probabilities(cls._base_classifier, vector) for vector in vectors
            ])
            if base_probabilities:
                base_detected, base_confidence = max(base_probabilities.items(), key=lambda item: item[1])
                base_ranked = {
                    label: round(float(prob), 4)
                    for label, prob in sorted(base_probabilities.items(), key=lambda item: item[1], reverse=True)
                }

        if probabilities:
            predicted, confidence = max(probabilities.items(), key=lambda item: item[1])
            ranked = {
                label: round(float(prob), 4)
                for label, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            }
            return predicted, float(confidence), ranked, base_detected, float(base_confidence), base_ranked

        predicted = str(cls._classifier.predict(vectors[0])[0])
        return predicted, 0.0, {}, base_detected, float(base_confidence), base_ranked

    @classmethod
    def _ending_tail_vectors(cls, audio: np.ndarray, sr: int) -> list[np.ndarray]:
        if len(audio) == 0:
            return []

        tail_samples = int(sr * 0.65)
        min_samples = int(sr * 0.12)
        end_points = [len(audio)]
        if len(audio) > tail_samples:
            end_points.extend([
                max(min_samples, int(len(audio) * 0.78)),
                max(min_samples, int(len(audio) * 0.90)),
            ])
        unique_end_points = sorted(set(end_points))

        vectors = []
        for end in unique_end_points:
            start = max(0, end - tail_samples)
            window = audio[start:end]
            if len(window) >= min_samples:
                vectors.append(cls._feature_vector(window, sr))
        return vectors or [cls._feature_vector(audio[-tail_samples:], sr)]

    @classmethod
    def _feature_vector(cls, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        inputs = cls._processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = cls._wav2vec2(**inputs).last_hidden_state[0].cpu().numpy()

        pieces = []
        for frac in (0.15, 0.25, 0.40, 0.65, 1.0):
            start = max(0, int(len(hidden) * (1.0 - frac)))
            window = hidden[start:] if start < len(hidden) else hidden
            pieces.extend([window.mean(axis=0), window.std(axis=0)])
        duration_feature = np.array([len(audio) / sr], dtype="float32")
        vector = np.concatenate([*pieces, duration_feature]).astype("float32")
        if _uses_augmented_classifier():
            vector = np.concatenate([vector, ending_acoustic_features(audio, sr)]).astype("float32")
        return vector.reshape(1, -1)

    @staticmethod
    def _probabilities(classifier, vector: np.ndarray) -> dict[str, float]:
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(vector)[0]
            labels = [str(x) for x in classifier.classes_]
            return {label: float(prob) for label, prob in zip(labels, proba)}
        predicted = str(classifier.predict(vector)[0])
        return {predicted: 1.0}


def _segment_by_reference_timing(audio: np.ndarray, sr: int, words: list[str]) -> list[tuple[int, int]]:
    """Fallback word segmentation when true forced alignment is unavailable.

    Arabic words in ETIB sentences can vary a lot in length. Equal splitting makes
    long words lose their ending and short words absorb neighbors, so we first find
    the spoken region, then allocate it by a rough pronunciation weight from the
    fully diacritized reference.
    """
    n_words = len(words)
    speech_start, speech_end = _speech_bounds(audio, sr)
    if speech_end <= speech_start or n_words <= 0:
        return _equal_segments(len(audio), n_words)

    weights = np.array([_word_timing_weight(word) for word in words], dtype="float32")
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return _segment_by_energy(audio, sr, n_words)

    total = speech_end - speech_start
    cumulative = np.concatenate([[0.0], np.cumsum(weights / weights.sum())])
    pad = int(sr * 0.035)
    return [
        _pad_interval(
            speech_start + int(total * cumulative[i]),
            speech_start + int(total * cumulative[i + 1]),
            len(audio),
            pad,
        )
        for i in range(n_words)
    ]


def _speech_bounds(audio: np.ndarray, sr: int) -> tuple[int, int]:
    if len(audio) == 0:
        return 0, 0

    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    energies = []
    starts = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        energies.append(float(np.sqrt(np.mean(audio[start:start + frame] ** 2))))
        starts.append(start)
    if not energies:
        return 0, len(audio)

    energies_arr = np.array(energies)
    threshold = max(float(np.percentile(energies_arr, 30)) * 1.05, 0.010)
    voiced = np.where(energies_arr > threshold)[0]
    if len(voiced) == 0:
        return 0, len(audio)

    start = max(0, starts[int(voiced[0])] - int(sr * 0.06))
    end_frame_start = starts[int(voiced[-1])]
    end = min(len(audio), end_frame_start + frame + int(sr * 0.08))
    return start, end


def _word_timing_weight(word: str) -> float:
    cleaned = _clean_word(word)
    letters = re.sub(r"[\u064b-\u0652\u0670]", "", cleaned)
    letters = re.sub(r"[^\u0621-\u064a]", "", letters)
    harakat = len(re.findall(r"[\u064b-\u0652]", cleaned))
    shadda = cleaned.count("\u0651")
    long_vowels = len(re.findall(r"[اويى]", letters))
    tanween = len(re.findall(r"[\u064b-\u064d]", cleaned))
    return max(0.7, len(letters) + 0.25 * harakat + 0.7 * shadda + 0.35 * long_vowels + 0.4 * tanween)


def _segment_by_energy(audio: np.ndarray, sr: int, n_words: int) -> list[tuple[int, int]]:
    if n_words <= 0:
        return []
    if len(audio) == 0:
        return [(0, 0)] * n_words

    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    energies = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        energies.append(float(np.sqrt(np.mean(audio[start:start + frame] ** 2))))
    if not energies:
        return _equal_segments(len(audio), n_words)

    energies_arr = np.array(energies)
    threshold = max(float(np.percentile(energies_arr, 35)) * 0.8, 0.012)
    voiced = energies_arr > threshold
    intervals = []
    start_frame: Optional[int] = None
    for i, active in enumerate(voiced):
        if active and start_frame is None:
            start_frame = i
        elif not active and start_frame is not None:
            intervals.append((start_frame * hop, min(len(audio), i * hop + frame)))
            start_frame = None
    if start_frame is not None:
        intervals.append((start_frame * hop, len(audio)))

    merged = []
    max_gap = int(sr * 0.12)
    for start, end in intervals:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    if len(merged) == n_words:
        return [_pad_interval(s, e, len(audio), int(sr * 0.04)) for s, e in merged]
    if len(merged) > n_words:
        return _merge_to_count(merged, n_words, len(audio), sr)
    if merged:
        return _equal_segments_between(merged[0][0], merged[-1][1], n_words, len(audio), int(sr * 0.03))
    return _equal_segments(len(audio), n_words)


def _candidate_endings(expected: str) -> list[str]:
    if expected in {"tanwin_fath", "tanwin_damm", "tanwin_kasr"}:
        return ["tanwin_fath", "tanwin_damm", "tanwin_kasr", "none"]
    if expected == "none":
        return ["none", "fatha", "damma", "kasra"]
    return ["fatha", "damma", "kasra", "none"]


def _apply_base_vowel_fallback(
    expected: str,
    detected: str,
    confidence: float,
    base_detected: str,
    base_confidence: float,
) -> tuple[str, float]:
    """Use the stronger coarse model when exact-class confidence is weak.

    The coarse model only decides final vowel family (a/u/i/none). For tanween,
    it can confirm the vowel family but cannot safely invent the final /n/.
    """
    if not base_detected or base_confidence < 0.62:
        return detected, confidence

    detected_base = _to_base_vowel(detected)
    expected_base = _to_base_vowel(expected)
    if detected_base == base_detected:
        return detected, max(confidence, min(base_confidence, 0.92))

    if confidence >= 0.70:
        return detected, confidence

    if base_detected == expected_base:
        return expected, min(base_confidence, 0.88)

    return base_detected, min(base_confidence, 0.88)


def _to_base_vowel(label: str) -> str:
    if label in {"damma", "tanwin_damm"}:
        return "damma"
    if label in {"fatha", "tanwin_fath"}:
        return "fatha"
    if label in {"kasra", "tanwin_kasr"}:
        return "kasra"
    return "none"


def _is_gradable_reference_word(word: str) -> bool:
    cleaned = _clean_word(word)
    if not cleaned:
        return False
    if re.search(r"[\u064b-\u0652]$", cleaned):
        return True
    # Accusative tanween is commonly written before final silent alif.
    return bool(cleaned[-1:] in {"ا", "ى"} and "\u064b" in cleaned[-5:])


def _wav2vec2_source() -> tuple[str, bool]:
    """Return local model folder when present, otherwise a Hugging Face model id."""
    if LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR), True
    return REMOTE_MODEL_ID, False

def _active_classifier_path() -> Path:
    if os.environ.get("USE_AUGMENTED_ACOUSTIC_CLASSIFIER") == "1" and AUGMENTED_CLASSIFIER_PATH.exists():
        return AUGMENTED_CLASSIFIER_PATH
    if CLASSIFIER_PATH.exists():
        return CLASSIFIER_PATH
    return FALLBACK_CLASSIFIER_PATH


def _uses_augmented_classifier() -> bool:
    return _active_classifier_path() == AUGMENTED_CLASSIFIER_PATH


def _combine_probabilities(
    primary: dict[str, float],
    fallback: dict[str, float],
    primary_weight: float,
) -> dict[str, float]:
    fallback_weight = 1.0 - primary_weight
    labels = set(primary) | set(fallback)
    combined = {
        label: primary.get(label, 0.0) * primary_weight + fallback.get(label, 0.0) * fallback_weight
        for label in labels
    }
    total = sum(combined.values())
    if total > 0:
        combined = {label: value / total for label, value in combined.items()}
    return combined


def _average_probability_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    items = [item for item in items if item]
    if not items:
        return {}
    labels = set().union(*(item.keys() for item in items))
    averaged = {
        label: float(np.mean([item.get(label, 0.0) for item in items]))
        for label in labels
    }
    total = sum(averaged.values())
    if total > 0:
        averaged = {label: value / total for label, value in averaged.items()}
    return averaged


def _align_timestamp_spans(
    reference_words: list[str],
    timestamp_words: list[dict],
    audio_len: int,
    sr: int,
) -> list[tuple[int, int]]:
    if not reference_words or not timestamp_words or not sr:
        return []

    ref_norm = [_normalize_for_alignment(word) for word in reference_words]
    hyp_norm = [_normalize_for_alignment(item.get("word", "")) for item in timestamp_words]
    hyp_norm = [word for word in hyp_norm if word]
    if not hyp_norm:
        return []

    spans_by_ref: list[Optional[tuple[int, int]]] = [None] * len(reference_words)
    used = set()
    search_from = 0

    for ref_i, ref_word in enumerate(ref_norm):
        if not ref_word:
            continue
        best_j = None
        best_score = 0.0
        for hyp_j in range(search_from, len(timestamp_words)):
            if hyp_j in used:
                continue
            hyp_word = _normalize_for_alignment(timestamp_words[hyp_j].get("word", ""))
            if not hyp_word:
                continue
            score = _word_similarity(ref_word, hyp_word)
            if score > best_score:
                best_score = score
                best_j = hyp_j
            if score >= 0.98:
                break
        if best_j is not None and best_score >= 0.72:
            item = timestamp_words[best_j]
            start = max(0, int(float(item["start"]) * sr) - int(sr * 0.05))
            end = min(audio_len, int(float(item["end"]) * sr) + int(sr * 0.08))
            if end > start:
                spans_by_ref[ref_i] = (start, end)
                used.add(best_j)
                search_from = max(search_from, best_j + 1)

    if sum(span is not None for span in spans_by_ref) < max(1, len(reference_words) // 2):
        return []

    return _fill_missing_timestamp_spans(spans_by_ref, audio_len, sr)


def _timestamp_source(timestamp_words: list[dict]) -> str:
    sources = {str(item.get("source", "")).strip() for item in timestamp_words if item.get("source")}
    if sources:
        return "+".join(sorted(sources))
    return "external_word_timestamps"


def _fill_missing_timestamp_spans(
    spans_by_ref: list[Optional[tuple[int, int]]],
    audio_len: int,
    sr: int,
) -> list[tuple[int, int]]:
    """Interpolate missing word spans between matched timestamp anchors."""
    n_words = len(spans_by_ref)
    if n_words == 0:
        return []

    filled: list[Optional[tuple[int, int]]] = list(spans_by_ref)
    anchors = [i for i, span in enumerate(filled) if span is not None]
    if not anchors:
        return _equal_segments(audio_len, n_words)

    # Fill before first matched word.
    first = anchors[0]
    if first > 0:
        end = max(1, filled[first][0])
        for i, span in enumerate(_equal_segments_between(0, end, first, audio_len, int(sr * 0.02))):
            filled[i] = span

    # Fill gaps between matched words.
    for left, right in zip(anchors, anchors[1:]):
        gap_count = right - left - 1
        if gap_count <= 0:
            continue
        start = filled[left][1]
        end = filled[right][0]
        if end <= start:
            start = filled[left][0]
            end = filled[right][1]
        for offset, span in enumerate(
            _equal_segments_between(start, end, gap_count, audio_len, int(sr * 0.02)),
            start=1,
        ):
            filled[left + offset] = span

    # Fill after last matched word.
    last = anchors[-1]
    if last < n_words - 1:
        start = min(audio_len - 1, filled[last][1])
        tail_count = n_words - last - 1
        for offset, span in enumerate(
            _equal_segments_between(start, audio_len, tail_count, audio_len, int(sr * 0.02)),
            start=1,
        ):
            filled[last + offset] = span

    return [
        span if span is not None else _equal_segments(audio_len, n_words)[i]
        for i, span in enumerate(filled)
    ]


def _normalize_for_alignment(word: str) -> str:
    word = _clean_word(word)
    word = re.sub(r"[\u064b-\u0652\u0670]", "", word)
    word = re.sub(r"[إأآٱ]", "ا", word)
    word = word.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    word = re.sub(r"[^\u0621-\u064a]", "", word)
    return word


def _word_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _merge_to_count(intervals: list[tuple[int, int]], n_words: int, total: int, sr: int) -> list[tuple[int, int]]:
    groups = []
    for i in range(n_words):
        start_idx = round(i * len(intervals) / n_words)
        end_idx = round((i + 1) * len(intervals) / n_words)
        group = intervals[start_idx:max(start_idx + 1, end_idx)]
        groups.append(_pad_interval(group[0][0], group[-1][1], total, int(sr * 0.04)))
    return groups


def _equal_segments(total: int, n_words: int) -> list[tuple[int, int]]:
    return [
        (int(i * total / n_words), int((i + 1) * total / n_words))
        for i in range(n_words)
    ]


def _equal_segments_between(start: int, end: int, n_words: int, total: int, pad: int) -> list[tuple[int, int]]:
    span = max(1, end - start)
    return [
        _pad_interval(
            start + int(i * span / n_words),
            start + int((i + 1) * span / n_words),
            total,
            pad,
        )
        for i in range(n_words)
    ]


def _pad_interval(start: int, end: int, total: int, pad: int) -> tuple[int, int]:
    return max(0, start - pad), min(total, end + pad)


def _clean_word(word: str) -> str:
    return re.sub(r"^[^\u0600-\u06FF]+|[^\u0600-\u06FF\u064b-\u0652]+$", "", word)


def strip_final_diacritics(word: str) -> str:
    return re.sub(r"[\u064b-\u0652]+$", "", word)


def _explain(word: str, expected: str, detected: str, status: str) -> Optional[str]:
    if status == "correct":
        return None
    expected_ar = ENDING_LABELS_AR.get(expected, expected)
    detected_ar = ENDING_LABELS_AR.get(detected, detected)
    return f"الكلمة '{word}' نهايتها المتوقعة {expected_ar}، لكن النموذج سمع {detected_ar}."
