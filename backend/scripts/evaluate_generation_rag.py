"""Offline Phase 3 RAG speech-generation evaluator.

This harness compares retrieval methods while keeping generation controls fixed.
It does not call Flask endpoints. The default generator is deterministic mock
output for structural pipeline validation only; real provider mode requires
explicit opt-in and preflight validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv


def resolve_repository_root(script_file: str | Path = __file__) -> Path:
    return Path(script_file).resolve().parents[2]


REPO_ROOT = resolve_repository_root()
BACKEND_ROOT = REPO_ROOT / 'backend'
PHASE3_FIXTURE_DIR = BACKEND_ROOT / 'tests' / 'fixtures' / 'rag_phase3'
PHASE3_CASES_PATH = PHASE3_FIXTURE_DIR / 'generation_cases.json'
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / 'reports' / 'rag_results' / 'phase3_generation_mock_results.json'

MAX_EVIDENCE_CHUNKS = 3
MAX_EVIDENCE_CHARACTERS = 3600
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 0
GENERATION_MODE_MOCK = 'mock'
GENERATION_MODE_REAL = 'real'
DENSE_MODE_STUB = 'stub'
DENSE_MODE_REAL = 'real'
DEFAULT_REAL_PROVIDER = 'gemini'
DEFAULT_REAL_TEMPERATURE = 0.0
DEFAULT_REAL_MAX_TOKENS = 2800
PROMPT_TEMPLATE_VERSION = 'phase3_canonical_prompt_v1'
SUPPORTED_REAL_PROVIDERS = {
    'gemini': {
        'required_env': 'GOOGLE_AI_KEY',
        'model': 'gemini-1.5-flash-latest',
    },
}


def load_evaluator_dotenv(repo_root: Path = REPO_ROOT) -> bool:
    dotenv_path = repo_root / '.env'
    if not dotenv_path.exists():
        return False
    load_dotenv(dotenv_path, override=False)
    return True


load_evaluator_dotenv()

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.document_grounding import chunk_text, normalize_text, _tokenize  # noqa: E402
from utils.embedding_retrieval import (  # noqa: E402
    DENSE_RETRIEVAL_METHOD,
    DEFAULT_DENSE_MODEL_NAME,
    DenseEmbeddingUnavailable,
    is_dense_embedding_available,
    select_relevant_chunks_dense_with_metadata,
)
from utils.graphrag_retrieval import (  # noqa: E402
    GRAPHRAG_RETRIEVAL_METHOD,
    select_relevant_chunks_graphrag_with_metadata,
)
from utils.lightrag_retrieval import (  # noqa: E402
    LIGHTRAG_RETRIEVAL_METHOD,
    select_relevant_chunks_lightrag_with_metadata,
)


RETRIEVAL_METHODS = {
    DENSE_RETRIEVAL_METHOD: {
        'selector': select_relevant_chunks_dense_with_metadata,
        'requires_optional_dense': True,
    },
    LIGHTRAG_RETRIEVAL_METHOD: {
        'selector': select_relevant_chunks_lightrag_with_metadata,
        'requires_optional_dense': False,
    },
    GRAPHRAG_RETRIEVAL_METHOD: {
        'selector': select_relevant_chunks_graphrag_with_metadata,
        'requires_optional_dense': False,
    },
}


class GenerationConfigError(RuntimeError):
    """Raised when real-generation controls are unsafe or incomplete."""


def load_generation_cases(cases_path: Path = PHASE3_CASES_PATH) -> list[dict]:
    return json.loads(cases_path.read_text(encoding='utf-8'))


def select_generation_cases(cases: list[dict], case_id: str | None = None) -> list[dict]:
    if not case_id:
        return cases
    selected = [case for case in cases if case.get('case_id') == case_id]
    if selected:
        return selected
    available = ', '.join(case.get('case_id', '<missing>') for case in cases)
    raise GenerationConfigError(
        f"Unknown case_id '{case_id}'. Available case IDs: {available}"
    )


def validate_case_schema(case: dict) -> None:
    required_fields = {
        'case_id',
        'source_documents',
        'user_request',
        'target_language',
        'requested_duration_seconds',
        'expected_source_documents',
        'key_factual_claims',
        'expected_relation_chain',
        'retrieval_type_category',
        'evaluator_notes',
    }
    missing = required_fields - set(case)
    if missing:
        raise ValueError(f"Case {case.get('case_id', '<unknown>')} missing fields: {sorted(missing)}")
    if not isinstance(case['source_documents'], list) or not case['source_documents']:
        raise ValueError(f"Case {case['case_id']} must list at least one source document")


def load_case_chunk_records(case: dict) -> list[dict]:
    records = []
    for source_order, filename in enumerate(case['source_documents']):
        path = PHASE3_FIXTURE_DIR / filename
        text = normalize_text(path.read_text(encoding='utf-8'))
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)):
            records.append({
                'text': chunk,
                'source_filename': filename,
                'source_type': '.txt',
                'chunk_index': chunk_index,
                'source_order': source_order,
            })
    return records


def deterministic_dense_stub_selector(
    chunk_records: list[dict],
    params: dict,
    max_excerpts: int,
    max_total_characters: int,
) -> list[dict]:
    """Deterministic dense stand-in for tests when embeddings are unavailable."""
    query_terms = _tokenize(' '.join(str(params.get(key) or '') for key in ('query', 'language')))
    scored = []
    for record in chunk_records:
        text_terms = _tokenize(record.get('text', ''))
        scored_record = dict(record)
        scored_record['score'] = len(query_terms.intersection(text_terms))
        scored_record['retrieval_method'] = DENSE_RETRIEVAL_METHOD
        scored.append(scored_record)

    scored.sort(key=lambda item: (
        -item.get('score', 0),
        item.get('source_order', 0),
        item.get('chunk_index', 0),
    ))
    return _select_scored_evidence(scored, max_excerpts, max_total_characters, DENSE_RETRIEVAL_METHOD)


def _select_scored_evidence(
    scored_chunks: list[dict],
    max_excerpts: int,
    max_total_characters: int,
    retrieval_method: str,
) -> list[dict]:
    selected = []
    total_chars = 0
    for chunk in scored_chunks:
        if len(selected) >= max_excerpts:
            break
        text_length = len(chunk.get('text', ''))
        if total_chars and total_chars + text_length > max_total_characters:
            continue
        selected.append({
            'text': chunk.get('text', ''),
            'source_filename': chunk.get('source_filename'),
            'source_type': chunk.get('source_type'),
            'chunk_index': chunk.get('chunk_index', 0),
            'score': chunk.get('score', 0),
            'retrieval_method': retrieval_method,
        })
        total_chars += text_length
    return selected


def retrieve_evidence(
    case: dict,
    method_name: str,
    dense_mode: str = DENSE_MODE_STUB,
) -> tuple[list[dict], float, str]:
    if method_name not in RETRIEVAL_METHODS:
        raise ValueError(f'Unknown retrieval method: {method_name}')
    validate_dense_mode(dense_mode)

    params = {
        'query': case['user_request'],
        'language': case['target_language'],
        'domain': case['retrieval_type_category'],
        'scenario': 'phase 3 offline generation evaluation',
    }
    records = load_case_chunk_records(case)
    method = RETRIEVAL_METHODS[method_name]
    selector = method['selector']
    runtime = 'real'

    if method_name == DENSE_RETRIEVAL_METHOD and dense_mode == DENSE_MODE_STUB:
        selector = deterministic_dense_stub_selector
        runtime = 'deterministic_stub'
    elif method_name == DENSE_RETRIEVAL_METHOD:
        ensure_real_dense_available()
        selector = select_relevant_chunks_dense_with_metadata

    started = time.perf_counter()
    try:
        def select():
            return selector(
                records,
                params,
                max_excerpts=MAX_EVIDENCE_CHUNKS,
                max_total_characters=MAX_EVIDENCE_CHARACTERS,
            )

        if method_name == DENSE_RETRIEVAL_METHOD and dense_mode == DENSE_MODE_REAL:
            evidence = _with_offline_dense_environment(select)
        else:
            evidence = select()
    except DenseEmbeddingUnavailable:
        if method_name == DENSE_RETRIEVAL_METHOD and dense_mode == DENSE_MODE_REAL:
            raise GenerationConfigError(
                'Real dense retrieval is unavailable and --dense-mode real forbids fallback. '
                f'Model: {DEFAULT_DENSE_MODEL_NAME}.'
            )
        evidence = deterministic_dense_stub_selector(
            records,
            params,
            max_excerpts=MAX_EVIDENCE_CHUNKS,
            max_total_characters=MAX_EVIDENCE_CHARACTERS,
        )
        runtime = 'deterministic_stub_dense_unavailable'
    latency_ms = (time.perf_counter() - started) * 1000
    return evidence, latency_ms, runtime


def build_canonical_prompt(case: dict, evidence_chunks: list[dict]) -> str:
    evidence_block = format_evidence(evidence_chunks)
    return f"""You are an expert ETIB interpreter-training speechwriter.

Task:
Generate a formal conference speech for interpreter training.

Controls:
- Target language: {case['target_language']}
- Requested duration: {case['requested_duration_seconds']} seconds
- User request: {case['user_request']}
- Use only the supplied evidence.
- Do not invent facts, names, dates, organizations, statistics, or causal links.
- If evidence is insufficient, stay general and do not add unsupported detail.
- Do not mention the retrieval method.

Evidence:
{evidence_block}

Output:
Return only the speech script text.
"""


def canonical_prompt_without_evidence(prompt: str) -> str:
    before, marker, _after = prompt.partition('Evidence:\n')
    if not marker:
        return prompt
    return before + 'Evidence:\n<EVIDENCE_PLACEHOLDER>\n\nOutput:\nReturn only the speech script text.\n'


def prompt_template_hash(prompt: str) -> str:
    template = f'{PROMPT_TEMPLATE_VERSION}\n{canonical_prompt_without_evidence(prompt)}'
    return hashlib.sha256(template.encode('utf-8')).hexdigest()[:16]


def format_evidence(evidence_chunks: list[dict]) -> str:
    if not evidence_chunks:
        return '[No evidence retrieved]'
    lines = []
    for index, chunk in enumerate(evidence_chunks, start=1):
        lines.append(
            f"[Evidence {index} | {chunk.get('source_filename')} | chunk {chunk.get('chunk_index', 0)}]\n"
            f"{chunk.get('text', '')}"
        )
    return '\n\n'.join(lines)


def mock_generation_function(prompt: str, case: dict, evidence_chunks: list[dict]) -> str:
    """Deterministic structural output; not a real speech-quality signal."""
    sources = ', '.join(sorted({chunk.get('source_filename', '') for chunk in evidence_chunks}))
    claims = '; '.join(case.get('key_factual_claims', [])[:3])
    return (
        f"[MOCK {case['target_language']} SPEECH] "
        f"Request: {case['user_request']} "
        f"Sources: {sources}. "
        f"Evidence terms: {claims}. "
        "This deterministic mock validates the evaluation pipeline only."
    )


def _configured_env_var(name: str) -> bool:
    value = os.getenv(name, '').strip()
    return bool(value and not value.startswith('your_'))


def normalize_gemini_model_id(model_id: str | None) -> str | None:
    if not model_id:
        return None
    normalized = model_id.strip()
    if normalized.startswith('models/'):
        normalized = normalized.removeprefix('models/')
    return normalized


def resolved_provider_model(provider: str, model_override: str | None = None) -> str | None:
    provider_config = SUPPORTED_REAL_PROVIDERS.get(provider)
    if not provider_config:
        return model_override
    return normalize_gemini_model_id(model_override) or provider_config.get('model')


def list_available_text_generation_models(provider: str) -> list[str]:
    load_evaluator_dotenv()
    validate_generation_controls(GENERATION_MODE_REAL, provider, DEFAULT_REAL_TEMPERATURE, DEFAULT_REAL_MAX_TOKENS)
    if provider != 'gemini':
        raise GenerationConfigError(f"Model discovery is not supported for provider '{provider}'.")

    import requests

    key = os.getenv(SUPPORTED_REAL_PROVIDERS[provider]['required_env'], '').strip()
    response = requests.get(
        'https://generativelanguage.googleapis.com/v1beta/models',
        params={'key': key},
        timeout=60,
    )
    if not response.ok:
        body = response.text.strip().replace('\r', ' ').replace('\n', ' ')[:300]
        raise GenerationConfigError(
            f'Gemini model listing failed with HTTP {response.status_code}: {body or "<empty>"}'
        )

    data = response.json()
    models = []
    for model in data.get('models', []):
        methods = set(model.get('supportedGenerationMethods', []))
        model_name = normalize_gemini_model_id(model.get('name'))
        if model_name and 'generateContent' in methods:
            models.append(model_name)
    return sorted(set(models))


def validate_model_override(provider: str, model_override: str | None) -> str | None:
    resolved_model = resolved_provider_model(provider, model_override)
    if provider == 'gemini' and model_override:
        available_models = list_available_text_generation_models(provider)
        if not available_models:
            raise GenerationConfigError('Gemini model listing returned no text-generation models.')
        if resolved_model not in available_models:
            available = ', '.join(available_models)
            raise GenerationConfigError(
                f"Gemini model '{resolved_model}' is not available for text generation. "
                f'Available text-generation models: {available}'
            )
    return resolved_model


def validate_generation_controls(
    generation_mode: str,
    provider: str | None,
    temperature: float,
    max_tokens: int,
) -> None:
    if generation_mode not in {GENERATION_MODE_MOCK, GENERATION_MODE_REAL}:
        raise GenerationConfigError(
            f"Unsupported generation mode '{generation_mode}'. Use 'mock' or 'real'."
        )
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise GenerationConfigError('max_tokens must be a positive integer.')
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise GenerationConfigError('temperature must be a non-negative number.')
    if generation_mode == GENERATION_MODE_REAL:
        if provider not in SUPPORTED_REAL_PROVIDERS:
            supported = ', '.join(sorted(SUPPORTED_REAL_PROVIDERS))
            raise GenerationConfigError(
                f"Unsupported real-generation provider '{provider}'. Supported providers: {supported}."
            )
        required_env = SUPPORTED_REAL_PROVIDERS[provider]['required_env']
        if not _configured_env_var(required_env):
            raise GenerationConfigError(
                f'{required_env} is required for provider {provider} but is not configured.'
            )


def validate_dense_mode(dense_mode: str) -> None:
    if dense_mode not in {DENSE_MODE_STUB, DENSE_MODE_REAL}:
        raise GenerationConfigError(
            f"Unsupported dense mode '{dense_mode}'. Use 'stub' or 'real'."
        )


def ensure_real_dense_available() -> None:
    if not is_dense_embedding_available():
        raise GenerationConfigError(
            'Real dense retrieval requires sentence-transformers, which is unavailable. '
            'Install optional RAG dependencies with backend/requirements-rag-optional.txt.'
        )


def _with_offline_dense_environment(callable_obj):
    previous = {
        'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
        'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
    }
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        return callable_obj()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_report_output_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path.resolve()
    return (REPO_ROOT / output_path).resolve()


def display_report_output_path(output_path: Path) -> str:
    resolved = resolve_report_output_path(output_path)
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def validate_output_path(output_path: Path) -> None:
    resolved = resolve_report_output_path(output_path)
    allowed_dir = (BACKEND_ROOT / 'reports' / 'rag_results').resolve()
    try:
        resolved.relative_to(allowed_dir)
    except ValueError as exc:
        raise GenerationConfigError(
            'Output path must be under backend/reports/rag_results/.'
        ) from exc
    if resolved.suffix.lower() != '.json':
        raise GenerationConfigError('Output path must be a .json file.')


def preflight_real_generation(
    method_names: list[str] | None,
    case_id: str | None,
    dense_mode: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
    model_override: str | None = None,
) -> dict:
    load_evaluator_dotenv()
    cases = select_generation_cases(load_generation_cases(), case_id)
    for case in cases:
        validate_case_schema(case)
    methods = method_names or list(RETRIEVAL_METHODS)
    unknown_methods = [method for method in methods if method not in RETRIEVAL_METHODS]
    if unknown_methods:
        raise GenerationConfigError(f'Unknown retrieval methods: {unknown_methods}')
    validate_generation_controls(GENERATION_MODE_REAL, provider, temperature, max_tokens)
    validate_dense_mode(dense_mode)
    validate_output_path(output_path)

    try:
        from services import llm_service
    except Exception as exc:
        raise GenerationConfigError('services.llm_service could not be imported.') from exc
    if not callable(getattr(llm_service, 'generate_text', None)):
        raise GenerationConfigError('services.llm_service.generate_text is not callable.')
    resolved_model = validate_model_override(provider, model_override)

    return {
        'provider': provider,
        'model': resolved_model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'dense_mode': dense_mode,
        'case_count': len(cases),
        'method_count': len(methods),
        'generations_per_case_method': 1,
        'expected_generation_count': len(cases) * len(methods),
        'output_path': display_report_output_path(output_path),
        'llm_called': False,
    }


def real_generation_function_factory(
    provider: str,
    temperature: float,
    max_tokens: int,
    model_override: str | None = None,
) -> Callable[[str, dict, list[dict]], str]:
    def generate(prompt: str, case: dict, evidence_chunks: list[dict]) -> str:
        from services import llm_service

        previous_provider = getattr(llm_service, 'LLM_PROVIDER', None)
        previous_gemini_model = getattr(llm_service, 'GEMINI_MODEL', None)
        try:
            llm_service.LLM_PROVIDER = provider
            if provider == 'gemini' and model_override:
                llm_service.GEMINI_MODEL = normalize_gemini_model_id(model_override)
            return llm_service.generate_text(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You generate grounded interpreter-training speeches from supplied evidence.',
                    },
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            if previous_provider is not None:
                llm_service.LLM_PROVIDER = previous_provider
            if previous_gemini_model is not None:
                llm_service.GEMINI_MODEL = previous_gemini_model

    return generate


def sanitize_generation_error(exc: Exception) -> str:
    text = str(exc).replace('\r', ' ').replace('\n', ' ').strip()
    return text[:500] or exc.__class__.__name__


def evaluate_case_method(
    case: dict,
    method_name: str,
    generation_function: Callable[[str, dict, list[dict]], str] = mock_generation_function,
    generation_mode: str = GENERATION_MODE_MOCK,
    dense_mode: str = DENSE_MODE_STUB,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_REAL_TEMPERATURE,
    max_tokens: int = DEFAULT_REAL_MAX_TOKENS,
) -> dict:
    validate_case_schema(case)
    evidence, retrieval_latency_ms, retrieval_runtime = retrieve_evidence(
        case,
        method_name,
        dense_mode=dense_mode,
    )
    prompt = build_canonical_prompt(case, evidence)
    started = time.perf_counter()
    generation_error = None
    generated_speech = ''
    try:
        generated_speech = generation_function(prompt, case, evidence)
    except Exception as exc:
        if generation_mode != GENERATION_MODE_REAL:
            raise
        generation_error = sanitize_generation_error(exc)
    generation_latency_ms = (time.perf_counter() - started) * 1000

    return {
        'case_id': case['case_id'],
        'retrieval_method': method_name,
        'generation_mode': generation_mode,
        'dense_mode': dense_mode,
        'provider': provider,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'prompt_template_version': PROMPT_TEMPLATE_VERSION,
        'prompt_template_hash': prompt_template_hash(prompt),
        'target_language': case['target_language'],
        'requested_duration_seconds': case['requested_duration_seconds'],
        'retrieved_evidence': evidence,
        'canonical_prompt': prompt,
        'generated_speech': generated_speech,
        'generation_error': generation_error,
        'retrieval_runtime': retrieval_runtime,
        'context_character_count': sum(len(chunk.get('text', '')) for chunk in evidence),
        'max_context_character_budget': MAX_EVIDENCE_CHARACTERS,
        'max_evidence_chunks': MAX_EVIDENCE_CHUNKS,
        'retrieved_source_documents': [chunk.get('source_filename') for chunk in evidence],
        'retrieval_latency_ms': retrieval_latency_ms,
        'generation_latency_ms': generation_latency_ms,
        'grounding_proxy': grounding_proxy(case, evidence),
        'traceability_proxy': traceability_proxy(case, evidence),
        'prompt_controls': {
            'target_language': case['target_language'],
            'requested_duration_seconds': case['requested_duration_seconds'],
            'user_request': case['user_request'],
            'template_without_evidence': canonical_prompt_without_evidence(prompt),
        },
    }


def grounding_proxy(case: dict, evidence_chunks: list[dict]) -> dict:
    combined_text = ' '.join(chunk.get('text', '') for chunk in evidence_chunks)
    source_set = set(chunk.get('source_filename') for chunk in evidence_chunks)
    expected_sources = case.get('expected_source_documents', [])
    source_hits = [source for source in expected_sources if source in source_set]
    claim_hits = matching_terms(combined_text, case.get('key_factual_claims', []))
    chain_hits = matching_terms(combined_text, case.get('expected_relation_chain', []))
    return {
        'expected_source_coverage': 0 if not expected_sources else len(source_hits) / len(expected_sources),
        'expected_source_hits': source_hits,
        'key_factual_claim_hits': claim_hits,
        'key_factual_claim_hit_rate': (
            0 if not case.get('key_factual_claims') else len(claim_hits) / len(case['key_factual_claims'])
        ),
        'relation_chain_hits': chain_hits,
        'relation_chain_hit_rate': (
            0 if not case.get('expected_relation_chain') else len(chain_hits) / len(case['expected_relation_chain'])
        ),
        'within_context_budget': sum(len(chunk.get('text', '')) for chunk in evidence_chunks) <= MAX_EVIDENCE_CHARACTERS,
    }


def traceability_proxy(case: dict, evidence_chunks: list[dict]) -> dict:
    trace_map = {}
    for term in case.get('key_factual_claims', []) + case.get('expected_relation_chain', []):
        term_tokens = _tokenize(term)
        trace_map[term] = [
            {
                'source_filename': chunk.get('source_filename'),
                'chunk_index': chunk.get('chunk_index', 0),
            }
            for chunk in evidence_chunks
            if term_tokens and term_tokens.issubset(_tokenize(chunk.get('text', '')))
        ]
    return {
        'source_traceability_map': trace_map,
        'terms_with_evidence': [term for term, hits in trace_map.items() if hits],
    }


def matching_terms(text: str, terms: list[str]) -> list[str]:
    text_tokens = _tokenize(text)
    hits = []
    for term in terms:
        term_tokens = _tokenize(term)
        if term_tokens and term_tokens.issubset(text_tokens):
            hits.append(term)
    return hits


def run_generation_evaluation(
    method_names: list[str] | None = None,
    case_id: str | None = None,
    generation_mode: str = GENERATION_MODE_MOCK,
    dense_mode: str = DENSE_MODE_STUB,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_REAL_TEMPERATURE,
    max_tokens: int = DEFAULT_REAL_MAX_TOKENS,
) -> dict:
    validate_generation_controls(generation_mode, provider, temperature, max_tokens)
    validate_dense_mode(dense_mode)
    cases = select_generation_cases(load_generation_cases(), case_id)
    methods = method_names or list(RETRIEVAL_METHODS)
    generation_function = mock_generation_function
    resolved_model = None
    if generation_mode == GENERATION_MODE_REAL:
        resolved_model = validate_model_override(provider or '', model)
        generation_function = real_generation_function_factory(
            provider=provider or '',
            temperature=temperature,
            max_tokens=max_tokens,
            model_override=model,
        )
    results = [
        evaluate_case_method(
            case,
            method_name,
            generation_function=generation_function,
            generation_mode=generation_mode,
            dense_mode=dense_mode,
            provider=provider if generation_mode == GENERATION_MODE_REAL else None,
            model=resolved_model if generation_mode == GENERATION_MODE_REAL else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for case in cases
        for method_name in methods
    ]
    return {
        'benchmark_name': 'phase3_end_to_end_speech_generation_rag',
        'generation_mode': generation_mode,
        'dense_mode': dense_mode,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'method_names': methods,
        'case_id_filter': case_id,
        'provider': provider if generation_mode == GENERATION_MODE_REAL else None,
        'model': resolved_model if generation_mode == GENERATION_MODE_REAL else None,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'case_count': len(cases),
        'result_count': len(results),
        'fixed_context_budget': {
            'max_evidence_chunks': MAX_EVIDENCE_CHUNKS,
            'max_evidence_characters': MAX_EVIDENCE_CHARACTERS,
        },
        'real_llm_called': generation_mode == GENERATION_MODE_REAL,
        'results': results,
        'summary': summarize_results(results),
        'notes': [
            (
                'Mock generation validates pipeline structure only.'
                if generation_mode == GENERATION_MODE_MOCK
                else 'Real generation results require human review before any quality claim.'
            ),
            'Proxy metrics measure retrieved context availability, not final speech quality.',
            (
                'No Flask endpoint, frontend, external LLM provider, or production router is used.'
                if generation_mode == GENERATION_MODE_MOCK
                else 'No Flask endpoint, frontend, or production router is used.'
            ),
        ],
    }


def evaluate_retrieval_only(
    method_names: list[str] | None = None,
    case_id: str | None = None,
    dense_mode: str = DENSE_MODE_STUB,
) -> dict:
    validate_dense_mode(dense_mode)
    cases = select_generation_cases(load_generation_cases(), case_id)
    methods = method_names or list(RETRIEVAL_METHODS)
    results = []
    for case in cases:
        validate_case_schema(case)
        for method_name in methods:
            evidence, retrieval_latency_ms, retrieval_runtime = retrieve_evidence(
                case,
                method_name,
                dense_mode=dense_mode,
            )
            grounding = grounding_proxy(case, evidence)
            results.append({
                'case_id': case['case_id'],
                'retrieval_method': method_name,
                'dense_mode': dense_mode,
                'retrieval_runtime': retrieval_runtime,
                'ranked_source_documents': [chunk.get('source_filename') for chunk in evidence],
                'chunk_count': len(evidence),
                'evidence_character_count': sum(len(chunk.get('text', '')) for chunk in evidence),
                'expected_source_coverage': grounding['expected_source_coverage'],
                'key_claim_hit_rate': grounding['key_factual_claim_hit_rate'],
                'relation_chain_hit_rate': grounding['relation_chain_hit_rate'],
                'relation_terms_present': bool(grounding['relation_chain_hits']),
                'retrieval_latency_ms': retrieval_latency_ms,
            })
    return {
        'benchmark_name': 'phase3_retrieval_only_inspection',
        'generation_mode': None,
        'dense_mode': dense_mode,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'case_count': len(cases),
        'method_names': methods,
        'result_count': len(results),
        'real_llm_called': False,
        'results': results,
    }


def summarize_results(results: list[dict]) -> dict:
    by_method = {}
    for method_name in sorted({result['retrieval_method'] for result in results}):
        method_results = [result for result in results if result['retrieval_method'] == method_name]
        by_method[method_name] = {
            'cases': len(method_results),
            'average_expected_source_coverage': _average(
                result['grounding_proxy']['expected_source_coverage']
                for result in method_results
            ),
            'average_key_claim_hit_rate': _average(
                result['grounding_proxy']['key_factual_claim_hit_rate']
                for result in method_results
            ),
            'average_relation_chain_hit_rate': _average(
                result['grounding_proxy']['relation_chain_hit_rate']
                for result in method_results
            ),
            'all_within_context_budget': all(
                result['grounding_proxy']['within_context_budget']
                for result in method_results
            ),
            'retrieval_runtimes': sorted({result['retrieval_runtime'] for result in method_results}),
        }
    return by_method


def _average(values) -> float:
    values = list(values)
    if not values:
        return 0
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description='Run Phase 3 RAG generation evaluation.')
    parser.add_argument('--method', action='append', choices=sorted(RETRIEVAL_METHODS))
    parser.add_argument('--case-id', default=None, help='Evaluate one exact Phase 3 case_id.')
    parser.add_argument('--generation-mode', choices=[GENERATION_MODE_MOCK, GENERATION_MODE_REAL], default=GENERATION_MODE_MOCK)
    parser.add_argument('--provider', default=DEFAULT_REAL_PROVIDER)
    parser.add_argument('--model', default=None, help='Optional model override for the selected provider.')
    parser.add_argument('--temperature', type=float, default=DEFAULT_REAL_TEMPERATURE)
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_REAL_MAX_TOKENS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--preflight', action='store_true', help='Validate real-generation setup without calling a provider.')
    parser.add_argument('--list-available-models', action='store_true', help='List text-generation models for the selected provider without generating.')
    parser.add_argument('--retrieval-only', action='store_true', help='Inspect retrieval evidence without generating text.')
    parser.add_argument('--dense-mode', choices=[DENSE_MODE_STUB, DENSE_MODE_REAL], default=DENSE_MODE_STUB)
    parser.add_argument('--allow-real-dense', action='store_true', help='Deprecated alias for --dense-mode real.')
    parser.add_argument('--write', action='store_true', help='Write JSON report to backend/reports/rag_results.')
    args = parser.parse_args()
    dense_mode = DENSE_MODE_REAL if args.allow_real_dense else args.dense_mode

    try:
        if args.list_available_models:
            models = list_available_text_generation_models(args.provider)
            if models:
                print('Available text-generation models:')
                for model in models:
                    print(f'- {model}')
            else:
                print('No text-generation models were returned.')
            return 0

        if args.preflight:
            preflight = preflight_real_generation(
                method_names=args.method,
                case_id=args.case_id,
                dense_mode=dense_mode,
                provider=args.provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_path=args.output,
                model_override=args.model,
            )
            print('Phase 3 real-generation preflight')
            print(f"Provider: {preflight['provider']}")
            print(f"Model: {preflight['model']}")
            print(f"Temperature: {preflight['temperature']}")
            print(f"Max tokens: {preflight['max_tokens']}")
            print(f"Dense mode: {preflight['dense_mode']}")
            print(
                f"Expected generations: {preflight['case_count']} cases x "
                f"{preflight['method_count']} retrieval methods x "
                f"{preflight['generations_per_case_method']} generation each = "
                f"{preflight['expected_generation_count']} generations"
            )
            print(f"Output: {preflight['output_path']}")
            print('LLM called: false')
            return 0

        if args.retrieval_only:
            report = evaluate_retrieval_only(
                method_names=args.method,
                case_id=args.case_id,
                dense_mode=dense_mode,
            )
        else:
            report = run_generation_evaluation(
                method_names=args.method,
                case_id=args.case_id,
                generation_mode=args.generation_mode,
                dense_mode=dense_mode,
                provider=args.provider if args.generation_mode == GENERATION_MODE_REAL else None,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
    except GenerationConfigError as exc:
        print(f'Configuration error: {exc}', file=sys.stderr)
        return 2

    print(f"Benchmark: {report['benchmark_name']}")
    print(f"Generation mode: {report['generation_mode']}")
    print(f"Dense mode: {report['dense_mode']}")
    if report['generation_mode'] == GENERATION_MODE_REAL:
        print(f"Provider: {report['provider']}")
        print(f"Model: {report['model']}")
        print(f"Temperature: {report['temperature']}")
        print(f"Max tokens: {report['max_tokens']}")
    print(f"Cases: {report['case_count']}")
    print(f"Results: {report['result_count']}")
    if args.retrieval_only:
        for result in report['results']:
            print()
            print(f"Case: {result['case_id']}")
            print(f"Method: {result['retrieval_method']}")
            print(f"Dense mode: {result['dense_mode']}")
            print(f"Ranked sources: {', '.join(result['ranked_source_documents'])}")
            print(f"Chunk count: {result['chunk_count']}")
            print(f"Evidence characters: {result['evidence_character_count']}")
            print(f"Expected-source coverage: {result['expected_source_coverage']:.2f}")
            print(f"Key-claim hit rate: {result['key_claim_hit_rate']:.2f}")
            print(f"Relation-chain hit rate: {result['relation_chain_hit_rate']:.2f}")
            print(f"Retrieval latency ms: {result['retrieval_latency_ms']:.2f}")
    else:
        for method_name, metrics in report['summary'].items():
            print()
            print(f"Method: {method_name}")
            print(f"Average expected-source coverage: {metrics['average_expected_source_coverage']:.2f}")
            print(f"Average key-claim hit rate: {metrics['average_key_claim_hit_rate']:.2f}")
            print(f"Average relation-chain hit rate: {metrics['average_relation_chain_hit_rate']:.2f}")
            print(f"Within context budget: {metrics['all_within_context_budget']}")
            print(f"Retrieval runtimes: {', '.join(metrics['retrieval_runtimes'])}")

    if args.write:
        try:
            validate_output_path(args.output)
        except GenerationConfigError as exc:
            print(f'Configuration error: {exc}', file=sys.stderr)
            return 2
        resolved_output = resolve_report_output_path(args.output)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f"\nReport: {display_report_output_path(args.output)}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
