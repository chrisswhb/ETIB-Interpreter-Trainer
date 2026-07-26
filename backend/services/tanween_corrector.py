"""LLM tanween/final-haraka correction adapter.

The expert recommendation was to use Gemma 4 after ASR, then later fine-tune
Gemma with labelled data. This adapter supports an OpenAI-compatible hosted
Gemma endpoint when available, and falls back to the already configured Groq
LLM so the pipeline can be tested immediately.
"""

import json
import logging
import os

from utils.arabizi import extract_ending_from_arabic, normalize_arabic_word

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMMA_MODEL = "google/gemma-4-12B-it"


def _build_prompt(transcript: str, reference_text: str | None = None) -> str:
    reference_block = ""
    if reference_text:
        reference_block = (
            "\nReference/context text, if useful:\n"
            f"{reference_text}\n"
        )

    return f"""You are correcting Arabic ASR output for an interpreter-training system.

Task:
- Return the Arabic transcript with full tashkeel.
- Focus especially on final word endings and tanween.
- Preserve the same word order and wording unless the ASR clearly made a minor spelling normalization.
- Do not invent extra words.
- Return strict JSON only.

JSON schema:
{{
  "diacritized_text": "Arabic text with tashkeel",
  "tanween_notes": [
    {{"word": "word", "ending": "fatha|damma|kasra|tanwin_fath|tanwin_damm|tanwin_kasr|none", "reason": "short reason"}}
  ]
}}
{reference_block}
ASR transcript:
{transcript}"""


def _safe_json_from_text(text: str) -> dict:
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
    return {"diacritized_text": content, "tanween_notes": []}


def _normalize_note_endings(result: dict, transcript: str) -> dict:
    """Make LLM note labels consistent with the actual returned tashkeel."""
    diacritized = result.get("diacritized_text") or transcript
    words_by_base = {}
    for word in diacritized.split():
        base = normalize_arabic_word(word)
        if base:
            words_by_base.setdefault(base, word)

    fixed_notes = []
    for note in result.get("tanween_notes") or []:
        if not isinstance(note, dict):
            continue
        word = str(note.get("word") or "").strip()
        base = normalize_arabic_word(word)
        diacritized_word = words_by_base.get(base)
        if diacritized_word:
            actual = extract_ending_from_arabic(diacritized_word)
            if actual != "none":
                note["ending"] = actual
                note["diacritized_word"] = diacritized_word
        fixed_notes.append(note)

    result["tanween_notes"] = fixed_notes
    return result


async def correct_tanween_with_llm(transcript: str, reference_text: str | None = None) -> dict:
    """Correct ASR transcript into a diacritized/tanween-aware hypothesis."""
    if not transcript.strip():
        return {"error": "empty transcript", "diacritized_text": "", "tanween_notes": []}

    gemma_endpoint = os.environ.get("GEMMA_OPENAI_BASE_URL", "").rstrip("/")
    gemma_api_key = os.environ.get("GEMMA_API_KEY", "")
    gemma_model = os.environ.get("GEMMA_MODEL_ID", DEFAULT_GEMMA_MODEL)
    if gemma_endpoint and gemma_api_key:
        return await _chat_completion(
            base_url=gemma_endpoint,
            api_key=gemma_api_key,
            model=gemma_model,
            transcript=transcript,
            reference_text=reference_text,
            provider="gemma_openai_compatible",
        )

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if groq_api_key:
        result = await _chat_completion(
            base_url=GROQ_BASE_URL,
            api_key=groq_api_key,
            model=GROQ_FALLBACK_MODEL,
            transcript=transcript,
            reference_text=reference_text,
            provider="groq_fallback_for_gemma_step",
        )
        result["warning"] = (
            "Gemma endpoint is not configured, so this used the existing Groq LLM "
            "as a temporary tanween-correction substitute."
        )
        return result

    return {
        "error": "No Gemma endpoint or GROQ_API_KEY configured for tanween correction",
        "diacritized_text": transcript,
        "tanween_notes": [],
        "provider": "none",
    }


async def _chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    transcript: str,
    reference_text: str | None,
    provider: str,
) -> dict:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=90) as client:
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
                            "content": "You produce strict JSON for Arabic tashkeel and tanween correction.",
                        },
                        {"role": "user", "content": _build_prompt(transcript, reference_text)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1400,
                    "response_format": {"type": "json_object"},
                },
            )
        if resp.status_code != 200:
            logger.warning("%s tanween correction failed %s: %s", provider, resp.status_code, resp.text[:300])
            return {
                "error": f"{provider} tanween correction failed: {resp.status_code}",
                "diacritized_text": transcript,
                "tanween_notes": [],
                "provider": provider,
                "model": model,
            }
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = _safe_json_from_text(content)
        parsed.setdefault("tanween_notes", [])
        parsed.setdefault("diacritized_text", transcript)
        parsed = _normalize_note_endings(parsed, transcript)
        parsed["provider"] = provider
        parsed["model"] = model
        return parsed
    except Exception as exc:
        logger.exception("%s tanween correction failed", provider)
        return {
            "error": str(exc),
            "diacritized_text": transcript,
            "tanween_notes": [],
            "provider": provider,
            "model": model,
        }
