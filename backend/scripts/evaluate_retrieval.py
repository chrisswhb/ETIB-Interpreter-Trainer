"""
Evaluate local retrieval baselines.

This script uses only local document_grounding helpers. It does not call Flask,
LLM providers, embedding models, vector stores, or the network.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
FIXTURE_DIR = BACKEND_ROOT / 'tests' / 'fixtures' / 'rag'
CASES_PATH = FIXTURE_DIR / 'retrieval_cases.json'
RELATION_FIXTURE_DIR = BACKEND_ROOT / 'tests' / 'fixtures' / 'rag_phase2'
RELATION_CASES_PATH = RELATION_FIXTURE_DIR / 'relation_cases.json'
REPORT_DIR = BACKEND_ROOT / 'reports' / 'rag_results'
DEFAULT_OUTPUT_PATH = REPORT_DIR / 'rag_lite_baseline.json'
PHASE2_DENSE_OUTPUT_PATH = REPORT_DIR / 'phase2_dense_multidocument_relations.json'
PHASE2_LIGHTRAG_OUTPUT_PATH = REPORT_DIR / 'phase2_lightrag_relation_graph.json'

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.document_grounding import (  # noqa: E402
    chunk_text,
    normalize_text,
    select_relevant_chunks_bm25_with_metadata,
    select_relevant_chunks_with_metadata,
    _tokenize,
)
from utils.embedding_retrieval import (  # noqa: E402
    DENSE_RETRIEVAL_METHOD,
    HYBRID_RETRIEVAL_METHOD,
    DenseEmbeddingUnavailable,
    OPTIONAL_DEPENDENCY_MESSAGE,
    is_dense_embedding_available,
    select_relevant_chunks_dense_with_metadata,
    select_relevant_chunks_hybrid_with_metadata,
)
from utils.lightrag_retrieval import (  # noqa: E402
    LIGHTRAG_RETRIEVAL_METHOD,
    select_relevant_chunks_lightrag_with_metadata,
)


METHOD_NAME = 'rag_lite_keyword_metadata'
BM25_METHOD_NAME = 'bm25_keyword'
DENSE_METHOD_NAME = DENSE_RETRIEVAL_METHOD
HYBRID_METHOD_NAME = HYBRID_RETRIEVAL_METHOD
HYBRID_BM25_WEIGHT = 0.5
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_WEIGHT_CONFIGS = {
    'hybrid_bm25_dense_0_2_0_8': (0.2, 0.8),
    'hybrid_bm25_dense_0_3_0_7': (0.3, 0.7),
    'hybrid_bm25_dense_0_4_0_6': (0.4, 0.6),
    HYBRID_METHOD_NAME: (HYBRID_BM25_WEIGHT, HYBRID_DENSE_WEIGHT),
}
MAX_EXCERPTS = 3
CHUNK_SIZE = 520
CHUNK_OVERLAP = 0
PHASE2_MAX_EXCERPTS = 3
PHASE2_CHUNK_SIZE = 1200
PHASE2_CHUNK_OVERLAP = 0
BASE_METHODS = {
    METHOD_NAME: {
        'selector': select_relevant_chunks_with_metadata,
        'output_path': REPORT_DIR / 'rag_lite_baseline.json',
        'notes': [
            'Keyword and metadata overlap baseline.',
            'No LLM calls, embeddings, vector database, or network calls are used.',
            'Placeholder semantic and cross-language cases are skipped for now.',
        ],
        'known_limitations': [
            'No stemming or synonym matching.',
            'No semantic paraphrase retrieval.',
            'No cross-language retrieval.',
            'Character-window chunking can split phrases across chunk boundaries.',
        ],
    },
    BM25_METHOD_NAME: {
        'selector': select_relevant_chunks_bm25_with_metadata,
        'output_path': REPORT_DIR / 'bm25_baseline.json',
        'notes': [
            'Lightweight BM25 keyword baseline using the shared normalized tokenizer.',
            'No LLM calls, embeddings, vector database, or network calls are used.',
            'Placeholder semantic and cross-language cases are skipped for now.',
        ],
        'known_limitations': [
            'No stemming or synonym matching.',
            'No semantic paraphrase retrieval.',
            'No cross-language retrieval.',
            'Small fixture sets can make BM25 metrics look stronger than production behavior.',
        ],
    },
    DENSE_METHOD_NAME: {
        'selector': select_relevant_chunks_dense_with_metadata,
        'output_path': REPORT_DIR / 'dense_multilingual_embedding_baseline.json',
        'optional': True,
        'availability_check': is_dense_embedding_available,
        'notes': [
            'Dense multilingual embedding retrieval baseline for offline evaluation.',
            'Uses sentence-transformers only when the optional dependency and model are available.',
            'No LLM calls, vector database, endpoint integration, or network calls are made by evaluator logic.',
        ],
        'known_limitations': [
            'Model availability depends on local dependencies and cached/downloadable model files.',
            'No hybrid lexical+dense reranking yet.',
            'No vector database persistence.',
            'No GraphRAG or relation-graph reasoning.',
        ],
    },
}

PHASE2_DENSE_METHOD = {
    'method_name': DENSE_METHOD_NAME,
    'selector': select_relevant_chunks_dense_with_metadata,
    'output_path': PHASE2_DENSE_OUTPUT_PATH,
    'optional': True,
    'availability_check': is_dense_embedding_available,
    'notes': [
        'Phase 2 relation-heavy multi-document benchmark.',
        'Dense multilingual retrieval is measured as the current production baseline.',
        'Scores are not averaged with Phase 1 because the retrieval goal is different.',
    ],
    'known_limitations': [
        'No LightRAG or GraphRAG implementation is included.',
        'Dense retrieval ranks chunks independently and does not build an explicit relation graph.',
        'A top-k result can contain relevant terms without proving a complete multi-hop relation.',
    ],
}
PHASE2_LIGHTRAG_METHOD = {
    'method_name': LIGHTRAG_RETRIEVAL_METHOD,
    'selector': select_relevant_chunks_lightrag_with_metadata,
    'output_path': PHASE2_LIGHTRAG_OUTPUT_PATH,
    'notes': [
        'Phase 2 relation-heavy multi-document benchmark.',
        'Offline LightRAG-style prototype using deterministic local graph signals.',
        'No official LightRAG package, LLM call, vector database, or endpoint integration is used.',
    ],
    'known_limitations': [
        'Entity and relation extraction are lightweight deterministic heuristics.',
        'No learned graph embeddings, community summaries, or LLM-generated relation extraction.',
        'GraphRAG is not implemented or tested.',
    ],
}
PHASE2_METHODS = {
    DENSE_METHOD_NAME: PHASE2_DENSE_METHOD,
    LIGHTRAG_RETRIEVAL_METHOD: PHASE2_LIGHTRAG_METHOD,
}


def _hybrid_method_config(method_name: str, bm25_weight: float, dense_weight: float) -> dict:
    return {
        'selector': select_relevant_chunks_hybrid_with_metadata,
        'output_path': REPORT_DIR / f'{method_name}_baseline.json',
        'optional': True,
        'availability_check': is_dense_embedding_available,
        'bm25_weight': bm25_weight,
        'dense_weight': dense_weight,
        'notes': [
            'Hybrid retrieval baseline combining normalized BM25 and dense multilingual scores.',
            f'Weights: BM25={bm25_weight:.2f}, dense={dense_weight:.2f}.',
            'No LLM calls, vector database, endpoint integration, or network calls are made by evaluator logic.',
        ],
        'known_limitations': [
            'Requires optional sentence-transformers dependencies and the dense model.',
            'Weights are fixed inside the evaluator script and are not tuned per category.',
            'No vector database persistence.',
            'No GraphRAG or relation-graph reasoning.',
        ],
    }


METHODS = dict(BASE_METHODS)
for hybrid_method_name, (bm25_weight, dense_weight) in HYBRID_WEIGHT_CONFIGS.items():
    METHODS[hybrid_method_name] = _hybrid_method_config(
        hybrid_method_name,
        bm25_weight,
        dense_weight,
    )


def load_cases(cases_path: Path = CASES_PATH) -> list[dict]:
    return json.loads(cases_path.read_text(encoding='utf-8'))


def load_relation_cases(cases_path: Path = RELATION_CASES_PATH) -> list[dict]:
    return json.loads(cases_path.read_text(encoding='utf-8'))


def load_chunk_records(case: dict) -> list[dict]:
    records = []
    for source_order, filename in enumerate(case['fixture_files']):
        path = FIXTURE_DIR / filename
        text = normalize_text(path.read_text(encoding='utf-8'))
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for chunk_index, chunk in enumerate(chunks):
            records.append({
                'text': chunk,
                'source_filename': filename,
                'source_type': '.txt',
                'chunk_index': chunk_index,
                'source_order': source_order,
            })
    return records


def load_relation_chunk_records(case: dict) -> list[dict]:
    records = []
    for source_order, filename in enumerate(case['fixture_files']):
        path = RELATION_FIXTURE_DIR / filename
        text = normalize_text(path.read_text(encoding='utf-8'))
        chunks = chunk_text(
            text,
            chunk_size=PHASE2_CHUNK_SIZE,
            overlap=PHASE2_CHUNK_OVERLAP,
        )
        for chunk_index, chunk in enumerate(chunks):
            records.append({
                'text': chunk,
                'source_filename': filename,
                'source_type': '.txt',
                'chunk_index': chunk_index,
                'source_order': source_order,
            })
    return records


def matching_term_count(text: str, expected_terms: list[str]) -> int:
    text_tokens = _tokenize(text)
    hits = 0
    for term in expected_terms:
        term_tokens = _tokenize(term)
        if term_tokens and term_tokens.issubset(text_tokens):
            hits += 1
    return hits


def matching_terms(text: str, expected_terms: list[str]) -> list[str]:
    text_tokens = _tokenize(text)
    hits = []
    for term in expected_terms:
        term_tokens = _tokenize(term)
        if term_tokens and term_tokens.issubset(text_tokens):
            hits.append(term)
    return hits


def first_relevant_rank(selected_chunks: list[dict], case: dict) -> int | None:
    expected_source = case.get('expected_source')
    expected_chunk_index = case.get('expected_chunk_index')
    expected_terms = case.get('expected_terms', [])

    for index, chunk in enumerate(selected_chunks, start=1):
        source_match = expected_source and chunk.get('source_filename') == expected_source
        chunk_index_match = (
            expected_chunk_index is not None
            and chunk.get('chunk_index') == expected_chunk_index
        )
        term_match = matching_term_count(chunk.get('text', ''), expected_terms) > 0
        if expected_chunk_index is not None:
            if source_match and chunk_index_match and term_match:
                return index
        elif expected_terms and term_match:
            return index
        elif not expected_terms and source_match:
            return index
    return None


def _summarize_subset(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {
            'evaluated_cases': 0,
            'passed_cases': 0,
            'failed_cases': 0,
            'recall_at_1': 0,
            'recall_at_3': 0,
            'mrr': 0,
            'exact_term_hit_rate': 0,
            'average_latency_ms': 0,
        }

    passed = sum(1 for result in results if result['recall_at_3_hit'])
    term_hit_total = sum(result['expected_term_hits'] for result in results)
    expected_term_total = sum(result['expected_term_count'] for result in results)

    return {
        'evaluated_cases': total,
        'passed_cases': passed,
        'failed_cases': total - passed,
        'recall_at_1': sum(1 for result in results if result['recall_at_1_hit']) / total,
        'recall_at_3': sum(1 for result in results if result['recall_at_3_hit']) / total,
        'mrr': sum(result['reciprocal_rank'] for result in results) / total,
        'exact_term_hit_rate': (
            0 if expected_term_total == 0 else term_hit_total / expected_term_total
        ),
        'average_latency_ms': sum(result['latency_ms'] for result in results) / total,
    }


def evaluate_case(case: dict, method_config: dict) -> dict:
    if case.get('skip'):
        return {
            'id': case['id'],
            'category': case['category'],
            'skipped': True,
            'skip_reason': case.get('skip_reason', 'Skipped by case definition.'),
        }

    params = dict(case.get('params', {}))
    params['query'] = case['query']

    started = time.perf_counter()
    records = load_chunk_records(case)
    selector = method_config['selector']
    selector_kwargs = {
        'max_excerpts': MAX_EXCERPTS,
        'max_total_characters': MAX_EXCERPTS * CHUNK_SIZE,
    }
    if selector is select_relevant_chunks_hybrid_with_metadata:
        selector_kwargs.update({
            'bm25_weight': method_config.get('bm25_weight', HYBRID_BM25_WEIGHT),
            'dense_weight': method_config.get('dense_weight', HYBRID_DENSE_WEIGHT),
        })
    selected_chunks = selector(records, params, **selector_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000

    rank = first_relevant_rank(selected_chunks, case)
    combined_text = ' '.join(chunk.get('text', '') for chunk in selected_chunks)
    term_hits = matching_term_count(combined_text, case.get('expected_terms', []))
    expected_count = len(case.get('expected_terms', []))

    return {
        'id': case['id'],
        'category': case['category'],
        'difficulty': case.get('difficulty', 'unspecified'),
        'skipped': False,
        'query': case['query'],
        'expected_source': case.get('expected_source'),
        'expected_chunk_index': case.get('expected_chunk_index'),
        'expected_best_methods': case.get('expected_best_methods', []),
        'expected_weak_methods': case.get('expected_weak_methods', []),
        'selected_sources': [chunk.get('source_filename') for chunk in selected_chunks],
        'selected_chunk_indexes': [chunk.get('chunk_index') for chunk in selected_chunks],
        'first_relevant_rank': rank,
        'recall_at_1_hit': rank == 1,
        'recall_at_3_hit': rank is not None and rank <= 3,
        'reciprocal_rank': 0 if rank is None else 1 / rank,
        'expected_term_hits': term_hits,
        'expected_term_count': expected_count,
        'exact_term_hit_rate': 0 if expected_count == 0 else term_hits / expected_count,
        'latency_ms': latency_ms,
        'selected_chunks': selected_chunks,
    }


def evaluate_relation_case(case: dict, method_config: dict) -> dict:
    params = dict(case.get('params', {}))
    params['query'] = case['query']

    started = time.perf_counter()
    records = load_relation_chunk_records(case)
    selected_chunks = method_config['selector'](
        records,
        params,
        max_excerpts=PHASE2_MAX_EXCERPTS,
        max_total_characters=PHASE2_MAX_EXCERPTS * PHASE2_CHUNK_SIZE,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    selected_sources = [chunk.get('source_filename') for chunk in selected_chunks]
    selected_source_set = set(selected_sources)
    required_sources = case.get('required_sources', [])
    required_source_hits = [
        source for source in required_sources
        if source in selected_source_set
    ]
    required_source_coverage = (
        0 if not required_sources else len(required_source_hits) / len(required_sources)
    )

    combined_text = ' '.join(chunk.get('text', '') for chunk in selected_chunks)
    relation_terms = case.get('expected_relation_terms', [])
    relation_term_hits = matching_terms(combined_text, relation_terms)
    relation_term_hit_rate = (
        0 if not relation_terms else len(relation_term_hits) / len(relation_terms)
    )

    relation_chain = case.get('relation_chain', [])
    relation_chain_hits = matching_terms(combined_text, relation_chain)
    relation_chain_coverage = (
        0 if not relation_chain else len(relation_chain_hits) / len(relation_chain)
    )

    evidence_terms = case.get('expected_evidence_terms', [])
    evidence_hits = matching_terms(combined_text, evidence_terms)

    if required_source_coverage == 1 and relation_chain_coverage == 1:
        coverage_status = 'covered'
    elif required_source_coverage > 0 or relation_chain_coverage > 0:
        coverage_status = 'partial'
    else:
        coverage_status = 'failed'

    return {
        'id': case['id'],
        'category': case['category'],
        'difficulty': case.get('difficulty', 'unspecified'),
        'query': case['query'],
        'multi_document_required': bool(case.get('multi_document_required')),
        'required_sources': required_sources,
        'selected_sources': selected_sources,
        'distinct_source_documents_top_k': len(selected_source_set),
        'required_source_hits': required_source_hits,
        'required_source_coverage': required_source_coverage,
        'multi_document_recall_at_3_hit': required_source_coverage == 1,
        'expected_relation_terms': relation_terms,
        'relation_term_hits': relation_term_hits,
        'relation_term_hit_rate': relation_term_hit_rate,
        'relation_chain': relation_chain,
        'relation_chain_hits': relation_chain_hits,
        'relation_chain_coverage': relation_chain_coverage,
        'expected_evidence_terms': evidence_terms,
        'evidence_term_hits': evidence_hits,
        'coverage_status': coverage_status,
        'latency_ms': latency_ms,
        'selected_chunks': selected_chunks,
    }


def summarize(results: list[dict]) -> dict:
    evaluated = [result for result in results if not result.get('skipped')]
    skipped = [result for result in results if result.get('skipped')]
    total = len(evaluated)

    if total == 0:
        return {
            'total_cases': len(results),
            'evaluated_cases': 0,
            'skipped_cases': len(skipped),
            'passed_cases': 0,
            'failed_cases': 0,
            'recall_at_1': 0,
            'recall_at_3': 0,
            'mrr': 0,
            'exact_term_hit_rate': 0,
            'average_latency_ms': 0,
            'metrics_by_category': {},
            'metrics_by_difficulty': {},
        }

    overall = _summarize_subset(evaluated)

    return {
        'total_cases': len(results),
        'evaluated_cases': total,
        'skipped_cases': len(skipped),
        'passed_cases': overall['passed_cases'],
        'failed_cases': overall['failed_cases'],
        'recall_at_1': overall['recall_at_1'],
        'recall_at_3': overall['recall_at_3'],
        'mrr': overall['mrr'],
        'exact_term_hit_rate': overall['exact_term_hit_rate'],
        'average_latency_ms': overall['average_latency_ms'],
        'metrics_by_category': _metrics_by_key(evaluated, 'category'),
        'metrics_by_difficulty': _metrics_by_key(evaluated, 'difficulty'),
    }


def summarize_relation_results(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {
            'total_cases': 0,
            'evaluated_cases': 0,
            'multi_document_recall_at_3': 0,
            'required_source_coverage': 0,
            'relation_term_hit_rate': 0,
            'relation_chain_coverage': 0,
            'average_distinct_source_documents_top_3': 0,
            'average_latency_ms': 0,
            'covered_cases': 0,
            'partial_cases': 0,
            'failed_cases': 0,
            'metrics_by_difficulty': {},
        }

    return {
        'total_cases': total,
        'evaluated_cases': total,
        'multi_document_recall_at_3': (
            sum(1 for result in results if result['multi_document_recall_at_3_hit'])
            / total
        ),
        'required_source_coverage': (
            sum(result['required_source_coverage'] for result in results) / total
        ),
        'relation_term_hit_rate': (
            sum(result['relation_term_hit_rate'] for result in results) / total
        ),
        'relation_chain_coverage': (
            sum(result['relation_chain_coverage'] for result in results) / total
        ),
        'average_distinct_source_documents_top_3': (
            sum(result['distinct_source_documents_top_k'] for result in results) / total
        ),
        'average_latency_ms': sum(result['latency_ms'] for result in results) / total,
        'covered_cases': sum(1 for result in results if result['coverage_status'] == 'covered'),
        'partial_cases': sum(1 for result in results if result['coverage_status'] == 'partial'),
        'failed_cases': sum(1 for result in results if result['coverage_status'] == 'failed'),
        'metrics_by_difficulty': _relation_metrics_by_key(results, 'difficulty'),
    }


def _summarize_relation_subset(results: list[dict]) -> dict:
    if not results:
        return {
            'evaluated_cases': 0,
            'multi_document_recall_at_3': 0,
            'required_source_coverage': 0,
            'relation_term_hit_rate': 0,
            'relation_chain_coverage': 0,
        }
    total = len(results)
    return {
        'evaluated_cases': total,
        'multi_document_recall_at_3': (
            sum(1 for result in results if result['multi_document_recall_at_3_hit'])
            / total
        ),
        'required_source_coverage': (
            sum(result['required_source_coverage'] for result in results) / total
        ),
        'relation_term_hit_rate': (
            sum(result['relation_term_hit_rate'] for result in results) / total
        ),
        'relation_chain_coverage': (
            sum(result['relation_chain_coverage'] for result in results) / total
        ),
    }


def _relation_metrics_by_key(results: list[dict], key: str) -> dict:
    values = sorted({result.get(key, 'unspecified') for result in results})
    return {
        value: _summarize_relation_subset([
            result for result in results
            if result.get(key, 'unspecified') == value
        ])
        for value in values
    }


def _metrics_by_key(results: list[dict], key: str) -> dict:
    values = sorted({result.get(key, 'unspecified') for result in results})
    return {
        value: _summarize_subset([
            result for result in results
            if result.get(key, 'unspecified') == value
        ])
        for value in values
    }


def identify_strengths_and_weaknesses(metrics: dict) -> dict:
    by_category = metrics.get('metrics_by_category', {})
    strong = [
        category for category, values in by_category.items()
        if values['recall_at_1'] >= 0.8
    ]
    weak = [
        category for category, values in by_category.items()
        if values['recall_at_3'] < 0.8
    ]
    mixed = [
        category for category, values in by_category.items()
        if category not in strong and category not in weak
    ]
    return {
        'strong_categories': strong,
        'mixed_categories': mixed,
        'weak_categories': weak,
    }


def unavailable_method_report(method_name: str, reason: str) -> dict:
    cases = load_cases()
    skipped_results = [
        {
            'id': case['id'],
            'category': case['category'],
            'difficulty': case.get('difficulty', 'unspecified'),
            'skipped': True,
            'skip_reason': reason,
        }
        for case in cases
    ]
    metrics = summarize(skipped_results)
    method_config = METHODS[method_name]

    return {
        'method_name': method_name,
        'method_available': False,
        'unavailable_reason': reason,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics,
        'per_method_strengths_and_weaknesses': identify_strengths_and_weaknesses(metrics),
        'per_case_results': skipped_results,
        'notes': method_config['notes'],
        'known_limitations': method_config['known_limitations'],
    }


def unavailable_phase2_relation_report(method_name: str, reason: str) -> dict:
    cases = load_relation_cases()
    method_config = PHASE2_METHODS[method_name]
    return {
        'benchmark_name': 'phase2_relation_heavy_multidocument',
        'method_name': method_name,
        'method_available': False,
        'unavailable_reason': reason,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': {
            'total_cases': len(cases),
            'evaluated_cases': 0,
            'multi_document_recall_at_3': 0,
            'required_source_coverage': 0,
            'relation_term_hit_rate': 0,
            'relation_chain_coverage': 0,
            'average_distinct_source_documents_top_3': 0,
            'average_latency_ms': 0,
            'covered_cases': 0,
            'partial_cases': 0,
            'failed_cases': 0,
            'metrics_by_difficulty': {},
        },
        'per_case_results': [
            {
                'id': case['id'],
                'category': case['category'],
                'difficulty': case.get('difficulty', 'unspecified'),
                'skipped': True,
                'skip_reason': reason,
            }
            for case in cases
        ],
        'notes': method_config['notes'],
        'known_limitations': method_config['known_limitations'],
    }


def run_evaluation(
    method_name: str = METHOD_NAME,
    write_output: bool = True,
    output_path: Path | None = None,
) -> dict:
    if method_name not in METHODS:
        raise ValueError(f'Unknown retrieval method: {method_name}')

    method_config = METHODS[method_name]
    if method_config.get('optional'):
        availability_check = method_config.get('availability_check')
        if availability_check and not availability_check():
            return unavailable_method_report(
                method_name,
                'Optional dense embedding dependency sentence-transformers is not installed. '
                f'{OPTIONAL_DEPENDENCY_MESSAGE}',
            )

    cases = load_cases()
    try:
        per_case_results = [evaluate_case(case, method_config) for case in cases]
    except DenseEmbeddingUnavailable as exc:
        return unavailable_method_report(method_name, str(exc))

    metrics = summarize(per_case_results)

    report = {
        'method_name': method_name,
        'method_available': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics,
        'per_method_strengths_and_weaknesses': identify_strengths_and_weaknesses(metrics),
        'per_case_results': per_case_results,
        'notes': method_config['notes'],
        'known_limitations': method_config['known_limitations'],
    }

    if write_output and report.get('method_available', True):
        output_path = output_path or method_config['output_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    return report


def run_phase2_relation_evaluation(
    method_name: str = DENSE_METHOD_NAME,
    write_output: bool = True,
) -> dict:
    if method_name not in PHASE2_METHODS:
        raise ValueError(f'Unknown Phase 2 retrieval method: {method_name}')

    method_config = PHASE2_METHODS[method_name]
    availability_check = method_config.get('availability_check')
    if availability_check and not availability_check():
        return unavailable_phase2_relation_report(
            method_name,
            'Optional dense embedding dependency sentence-transformers is not installed. '
            f'{OPTIONAL_DEPENDENCY_MESSAGE}',
        )

    cases = load_relation_cases()
    try:
        per_case_results = [
            evaluate_relation_case(case, method_config)
            for case in cases
        ]
    except DenseEmbeddingUnavailable as exc:
        return unavailable_phase2_relation_report(method_name, str(exc))

    metrics = summarize_relation_results(per_case_results)
    report = {
        'benchmark_name': 'phase2_relation_heavy_multidocument',
        'method_name': method_name,
        'method_available': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics,
        'per_case_results': per_case_results,
        'notes': method_config['notes'],
        'known_limitations': method_config['known_limitations'],
    }

    if write_output:
        output_path = method_config['output_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    return report


def run_all_phase2_relation_evaluations(write_output: bool = True) -> list[dict]:
    return [
        run_phase2_relation_evaluation(method_name=method_name, write_output=write_output)
        for method_name in PHASE2_METHODS
    ]


def run_all_evaluations(write_output: bool = True) -> list[dict]:
    return [
        run_evaluation(method_name=method_name, write_output=write_output)
        for method_name in METHODS
    ]


def print_summary(report: dict) -> None:
    metrics = report['metrics']
    strengths = report['per_method_strengths_and_weaknesses']
    output_path = METHODS[report['method_name']]['output_path']
    print(f"Method: {report['method_name']}")
    if not report.get('method_available', True):
        print('Status: unavailable/skipped')
        print(f"Reason: {report.get('unavailable_reason', 'Unavailable')}")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Evaluated cases: {metrics['evaluated_cases']}")
    print(f"Skipped cases: {metrics['skipped_cases']}")
    print(f"Passed cases: {metrics['passed_cases']}")
    print(f"Failed cases: {metrics['failed_cases']}")
    print(f"Recall@1: {metrics['recall_at_1']:.2f}")
    print(f"Recall@3: {metrics['recall_at_3']:.2f}")
    print(f"MRR: {metrics['mrr']:.2f}")
    print(f"Exact term hit rate: {metrics['exact_term_hit_rate']:.2f}")
    print(f"Average latency: {metrics['average_latency_ms']:.2f} ms")
    print(f"Strong categories: {', '.join(strengths['strong_categories']) or 'none'}")
    print(f"Mixed categories: {', '.join(strengths['mixed_categories']) or 'none'}")
    print(f"Weak categories: {', '.join(strengths['weak_categories']) or 'none'}")
    if report.get('method_available', True):
        print(f"Report: {output_path.relative_to(REPO_ROOT)}")
    else:
        print('Report: not written because method is unavailable')


def print_phase2_relation_summary(report: dict) -> None:
    metrics = report['metrics']
    print('Benchmark: Phase 2 relation-heavy multi-document retrieval')
    print(f"Method: {report['method_name']}")
    if not report.get('method_available', True):
        print('Status: unavailable/skipped')
        print(f"Reason: {report.get('unavailable_reason', 'Unavailable')}")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Evaluated cases: {metrics['evaluated_cases']}")
    print(f"Multi-document Recall@3: {metrics['multi_document_recall_at_3']:.2f}")
    print(f"Required-source coverage: {metrics['required_source_coverage']:.2f}")
    print(f"Relation-term hit rate: {metrics['relation_term_hit_rate']:.2f}")
    print(f"Relation-chain coverage: {metrics['relation_chain_coverage']:.2f}")
    print(
        'Average distinct source documents in top 3: '
        f"{metrics['average_distinct_source_documents_top_3']:.2f}"
    )
    print(f"Covered cases: {metrics['covered_cases']}")
    print(f"Partial cases: {metrics['partial_cases']}")
    print(f"Failed cases: {metrics['failed_cases']}")
    print(f"Average latency: {metrics['average_latency_ms']:.2f} ms")
    if report.get('method_available', True):
        output_path = PHASE2_METHODS[report['method_name']]['output_path']
        print(f"Report: {output_path.relative_to(REPO_ROOT)}")
        partial_or_failed = [
            result['id'] for result in report['per_case_results']
            if result['coverage_status'] != 'covered'
        ]
        print(f"Partial/failed cases: {', '.join(partial_or_failed) or 'none'}")
    else:
        print('Report: not written because method is unavailable')


def main() -> int:
    reports = run_all_evaluations(write_output=True)
    for index, report in enumerate(reports):
        if index:
            print()
        print_summary(report)
    for report in run_all_phase2_relation_evaluations(write_output=True):
        print()
        print_phase2_relation_summary(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
