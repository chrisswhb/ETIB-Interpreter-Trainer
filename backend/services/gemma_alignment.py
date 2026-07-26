"""Gemma/LLM timestamp alignment refinement.

Gemma cannot hear audio directly in this setup. The useful version of the
expert's suggestion is:

1. get rough word timestamps from CTC/Whisper,
2. give Gemma the reference words plus rough timestamped words,
3. ask it to map each reference word to a start/end time.
"""

from __future__ import annotations

import json
import logging
import os
import re

from utils.arabizi import normalize_arabic_word

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMMA_MODEL = "google/gemma-4-12B-it"


async def refine_alignment_with_gemma(reference_sentence: str, rough_spans: list[dict]) -> list[dict]:
    if os.environ.get("ENABLE_GEMMA_ALIGNMENT") != "1":
        return []
    if not reference_sentence.strip() or not rough_spans:
        return []

    gemma_endpoint = os.environ.get("GEMMA_OPENAI_BASE_URL", "").rstrip("/")
    gemma_api_key = os.environ.get("GEMMA_API_KEY", "")
    gemma_model = os.environ.get("GEMMA_MODEL_ID", DEFAULT_GEMMA_MODEL)
    if gemma_endpoint and gemma_api_key:
        return await _chat_align(
            base_url=gemma_endpoint,
            api_key=gemma_api_key,
            model=gemma_model,
            provider="gemma_openai_compatible_alignment",
            reference_sentence=reference_sentence,
            rough_spans=rough_spans,
        )

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if groq_api_key and os.environ.get("ENABLE_GROQ_AS_GEMMA_ALIGNMENT") == "1":
        return await _chat_align(
            base_url=GROQ_BASE_URL,
            api_key=groq_api_key,
            model=GROQ_FALLBACK_MODEL,
            provider="groq_fallback_alignment",
            reference_sentence=reference_sentence,
            rough_spans=rough_spans,
        )

    return []


async def _chat_align(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    reference_sentence: str,
    rough_spans: list[dict],
) -> list[dict]:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You align Arabic reference words to rough ASR/CTC word timestamps. Return strict JSON only.",
                        },
                        {"role": "user", "content": _build_prompt(reference_sentence, rough_spans)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2400,
                    "response_format": {"type": "json_object"},
                },
            )
        if resp.status_code != 200:
            logger.warning("%s failed %s: %s", provider, resp.status_code, resp.text[:300])
            return []

        payload = _safe_json(resp.json()["choices"][0]["message"]["content"])
        aligned = payload.get("aligned_words") or []
        return _validate_alignment(reference_sentence, aligned, rough_spans, provider, model)
    except Exception as exc:
        logger.warning("%s unavailable: %s", provider, exc)
        return []


def _build_prompt(reference_sentence: str, rough_spans: list[dict]) -> str:
    reference_words = [_clean_word(w) for w in reference_sentence.split()]
    reference_words = [w for w in reference_words if w]
    compact_spans = [
        {
            "index": i,
            "word": str(item.get("word", "")),
            "start": round(float(item.get("start", 0.0)), 3),
            "end": round(float(item.get("end", 0.0)), 3),
        }
        for i, item in enumerate(rough_spans)
        if item.get("word") and item.get("start") is not None and item.get("end") is not None
    ]
    return f"""Task:
Map each Arabic reference word to the best rough timestamp interval.

Rules:
- The reference sentence is authoritative.
- Do not change the reference words.
- Use rough timestamps only as timing evidence.
- If one rough word corresponds to multiple reference words, split its time proportionally.
- If a reference word is missing in rough timestamps, interpolate from neighbors.
- Times are seconds.
- Return strict JSON only.

JSON schema:
{{
  "aligned_words": [
    {{"word_index": 0, "word": "reference word", "start": 0.0, "end": 0.5, "confidence": 0.0}}
  ]
}}

Reference words:
{json.dumps(reference_words, ensure_ascii=False)}

Rough timestamped words:
{json.dumps(compact_spans, ensure_ascii=False)}
"""


def _safe_json(text: str) -> dict:
    content = (text or "").strip().strip("`").strip()
    if content.startswith("json"):
        content = content[4:].strip()
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass
    return {}


def _validate_alignment(
    reference_sentence: str,
    aligned: list,
    rough_spans: list[dict],
    provider: str,
    model: str,
) -> list[dict]:
    reference_words = [_clean_word(w) for w in reference_sentence.split()]
    reference_words = [w for w in reference_words if w]
    if not isinstance(aligned, list):
        return []

    max_end = max((float(item.get("end", 0.0)) for item in rough_spans if item.get("end") is not None), default=0.0)
    by_index = {}
    for item in aligned:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("word_index"))
            start = max(0.0, float(item.get("start")))
            end = min(max_end or float(item.get("end")), float(item.get("end")))
        except Exception:
            continue
        if idx < 0 or idx >= len(reference_words) or end <= start:
            continue
        word = str(item.get("word") or reference_words[idx]).strip()
        if normalize_arabic_word(word) != normalize_arabic_word(reference_words[idx]):
            word = reference_words[idx]
        by_index[idx] = {
            "word": word,
            "start": start,
            "end": end,
            "source": provider,
            "model": model,
            "confidence": float(item.get("confidence") or 0.0),
        }

    if len(by_index) < max(1, len(reference_words) // 2):
        return []
    return [by_index[i] for i in sorted(by_index)]


def _clean_word(word: str) -> str:
    return re.sub(r"^[^\u0600-\u06FF]+|[^\u0600-\u06FF\u064b-\u0652]+$", "", word)
