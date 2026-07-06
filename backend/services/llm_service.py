"""
LLM provider abstraction.

Module code should call generate_text() instead of binding directly to a
specific cloud or local model client.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import (
    GOOGLE_AI_KEY,
    LLM_PROVIDER,
    LOCAL_MODEL_DEVICE_MAP,
    LOCAL_MODEL_ID,
    LOCAL_MODEL_PATH,
    LOCAL_MODEL_TORCH_DTYPE,
    PRIMARY_LLM_MODEL,
    REMOTE_AYA_TIMEOUT_SECONDS,
    REMOTE_AYA_URL,
    REMOTE_AYA_VERIFY_SSL,
)

# All clients are lazy — nothing is imported or created until first call.
_local_pipeline: Any | None = None
_local_tokenizer: Any | None = None


def _active_groq_key() -> str | None:
    """Return the per-request user key only."""
    try:
        from flask import g
        return getattr(g, 'groq_api_key', None)
    except RuntimeError:
        return None

GEMINI_MODEL = 'gemini-1.5-flash-latest'
MAX_GEMINI_THINKING_BUDGET = 2048


def _generation_result(
    text: str,
    *,
    provider: str,
    model: str | None,
    finish_reason: str | None = None,
    usage_metadata: dict[str, Any] | None = None,
    safety_metadata: list[dict[str, Any]] | None = None,
    candidate_count: int | None = None,
    response_metadata_available: bool = False,
) -> dict[str, Any]:
    return {
        'text': text,
        'provider': provider,
        'model': model,
        'finish_reason': finish_reason,
        'usage_metadata': usage_metadata,
        'safety_metadata': safety_metadata,
        'candidate_count': candidate_count,
        'response_metadata_available': response_metadata_available,
    }


def generate_text(
    messages: list[dict[str, str]],
    max_tokens: int = 1800,
    temperature: float = 0.7,
    return_metadata: bool = False,
    thinking_budget: int | None = None,
) -> str | dict[str, Any]:
    """Generate text using the configured provider."""
    _validate_thinking_budget(thinking_budget)
    provider = LLM_PROVIDER.lower().strip()

    if provider == 'gemini':
        return _generate_with_gemini(
            messages,
            max_tokens,
            temperature,
            return_metadata,
            thinking_budget,
        )

    if provider == 'groq':
        text = _generate_with_groq(messages, max_tokens, temperature)
        return _generation_result(text, provider='groq', model=PRIMARY_LLM_MODEL) if return_metadata else text

    if provider == 'local_aya':
        text = _generate_with_local_aya(messages, max_tokens, temperature)
        return _generation_result(text, provider='local_aya', model=LOCAL_MODEL_ID) if return_metadata else text

    if provider == 'remote_aya':
        text = _generate_with_remote_aya(messages, max_tokens, temperature)
        return _generation_result(text, provider='remote_aya', model=None) if return_metadata else text

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini', 'groq', 'local_aya', or 'remote_aya'."
    )


# ── Gemini — raw REST (no SDK, no httpx conflicts) ───────────────────────────

def _generate_with_gemini(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    return_metadata: bool = False,
    thinking_budget: int | None = None,
) -> str | dict[str, Any]:
    """Call Gemini via the REST API using requests — no google-genai SDK needed."""
    import os
    key = os.getenv('GOOGLE_AI_KEY', '').strip()
    if not key or key == 'your_google_ai_key_here':
        raise RuntimeError(
            'GOOGLE_AI_KEY is not configured. '
            'Get a free key at https://aistudio.google.com and add it to backend/.env'
        )

    import requests

    system_text = ''
    user_text = ''
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            system_text = content
        elif role == 'user':
            user_text = content

    # Prepend system instruction directly into the user turn — works on all API versions
    combined = f"{system_text}\n\n{user_text}".strip() if system_text else user_text

    payload: dict[str, Any] = {
        'contents': [{'role': 'user', 'parts': [{'text': combined}]}],
        'generationConfig': {
            'maxOutputTokens': max_tokens,
            'temperature': temperature,
        },
    }
    if thinking_budget is not None:
        payload['generationConfig']['thinkingConfig'] = {
            'thinkingBudget': thinking_budget,
        }

    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{GEMINI_MODEL}:generateContent?key={key}'
    )

    resp = requests.post(url, json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(
            f'Gemini API error {resp.status_code}: {resp.text[:400]}'
        )

    data = resp.json()
    try:
        candidate = data['candidates'][0]
        text = candidate['content']['parts'][0]['text'].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f'Unexpected Gemini response shape: {data}') from exc
    if not return_metadata:
        return text
    return _generation_result(
        text,
        provider='gemini',
        model=GEMINI_MODEL,
        finish_reason=candidate.get('finishReason'),
        usage_metadata=data.get('usageMetadata'),
        safety_metadata=candidate.get('safetyRatings'),
        candidate_count=len(data.get('candidates', [])),
        response_metadata_available=True,
    )


def _validate_thinking_budget(thinking_budget: int | None) -> None:
    if thinking_budget is None:
        return
    if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool):
        raise ValueError('thinking_budget must be a positive integer or None.')
    if thinking_budget <= 0 or thinking_budget > MAX_GEMINI_THINKING_BUDGET:
        raise ValueError(
            f'thinking_budget must be between 1 and {MAX_GEMINI_THINKING_BUDGET}.'
        )


# ── Groq ─────────────────────────────────────────────────────────────────────

def _generate_with_groq(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    global _groq_client

    key = _active_groq_key()
    if not key:
        raise RuntimeError(
            'No Groq API key found. Please open Settings and add your '
            'personal Groq API key (free at console.groq.com).'
        )

    from groq import Groq
    client = Groq(api_key=key)

    response = client.chat.completions.create(
        model=PRIMARY_LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ── Local Aya ─────────────────────────────────────────────────────────────────

def _generate_with_local_aya(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    global _local_pipeline, _local_tokenizer

    if _local_pipeline is None or _local_tokenizer is None:
        _local_pipeline, _local_tokenizer = _load_local_aya()

    prompt = _messages_to_prompt(messages, _local_tokenizer)
    outputs = _local_pipeline(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        return_full_text=False,
    )
    return outputs[0]['generated_text'].strip()


# ── Remote Aya ────────────────────────────────────────────────────────────────

def _generate_with_remote_aya(
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    if not REMOTE_AYA_URL:
        raise RuntimeError('REMOTE_AYA_URL is not configured.')

    import requests

    response = requests.post(
        REMOTE_AYA_URL,
        headers={'ngrok-skip-browser-warning': 'true'},
        json={'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature},
        timeout=REMOTE_AYA_TIMEOUT_SECONDS,
        verify=REMOTE_AYA_VERIFY_SSL,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()[:500]
        raise RuntimeError(
            f'remote_aya HTTP {response.status_code}: {body or "<empty>"}'
        ) from exc

    data = response.json()
    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError('remote_aya returned an empty or invalid response')
    return text.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_local_aya() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            'Local Aya requires torch, transformers, accelerate, sentencepiece, '
            'and huggingface-hub.'
        ) from exc

    model_ref = LOCAL_MODEL_PATH or LOCAL_MODEL_ID
    if LOCAL_MODEL_PATH and not Path(LOCAL_MODEL_PATH).exists():
        raise RuntimeError(f'LOCAL_MODEL_PATH does not exist: {LOCAL_MODEL_PATH}')

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        'device_map': LOCAL_MODEL_DEVICE_MAP,
        'trust_remote_code': True,
    }
    if LOCAL_MODEL_TORCH_DTYPE == 'auto':
        model_kwargs['torch_dtype'] = 'auto'
    elif hasattr(__import__('torch'), LOCAL_MODEL_TORCH_DTYPE):
        model_kwargs['torch_dtype'] = getattr(__import__('torch'), LOCAL_MODEL_TORCH_DTYPE)

    model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    return pipeline('text-generation', model=model, tokenizer=tokenizer), tokenizer


def _messages_to_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    parts = []
    for message in messages:
        role = message.get('role', 'user').upper()
        content = message.get('content', '')
        parts.append(f'{role}:\n{content}')
    parts.append('ASSISTANT:\n')
    return '\n\n'.join(parts)
