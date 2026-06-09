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
REPORT_DIR = BACKEND_ROOT / 'reports' / 'rag_results'
DEFAULT_OUTPUT_PATH = REPORT_DIR / 'rag_lite_baseline.json'

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils.document_grounding import (  # noqa: E402
    chunk_text,
    normalize_text,
    select_relevant_chunks_bm25_with_metadata,
    select_relevant_chunks_with_metadata,
    _tokenize,
)


METHOD_NAME = 'rag_lite_keyword_metadata'
BM25_METHOD_NAME = 'bm25_keyword'
MAX_EXCERPTS = 3
CHUNK_SIZE = 520
CHUNK_OVERLAP = 0
METHODS = {
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
}


def load_cases(cases_path: Path = CASES_PATH) -> list[dict]:
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


def matching_term_count(text: str, expected_terms: list[str]) -> int:
    text_tokens = _tokenize(text)
    hits = 0
    for term in expected_terms:
        term_tokens = _tokenize(term)
        if term_tokens and term_tokens.issubset(text_tokens):
            hits += 1
    return hits


def first_relevant_rank(selected_chunks: list[dict], case: dict) -> int | None:
    expected_source = case.get('expected_source')
    expected_terms = case.get('expected_terms', [])

    for index, chunk in enumerate(selected_chunks, start=1):
        source_match = expected_source and chunk.get('source_filename') == expected_source
        term_match = matching_term_count(chunk.get('text', ''), expected_terms) > 0
        if source_match or term_match:
            return index
    return None


def evaluate_case(case: dict, selector) -> dict:
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
    selected_chunks = selector(
        records,
        params,
        max_excerpts=MAX_EXCERPTS,
        max_total_characters=MAX_EXCERPTS * CHUNK_SIZE,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    rank = first_relevant_rank(selected_chunks, case)
    combined_text = ' '.join(chunk.get('text', '') for chunk in selected_chunks)
    term_hits = matching_term_count(combined_text, case.get('expected_terms', []))
    expected_count = len(case.get('expected_terms', []))

    return {
        'id': case['id'],
        'category': case['category'],
        'skipped': False,
        'query': case['query'],
        'expected_source': case.get('expected_source'),
        'selected_sources': [chunk.get('source_filename') for chunk in selected_chunks],
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
        }

    passed = sum(1 for result in evaluated if result['recall_at_3_hit'])
    term_hit_total = sum(result['expected_term_hits'] for result in evaluated)
    expected_term_total = sum(result['expected_term_count'] for result in evaluated)

    return {
        'total_cases': len(results),
        'evaluated_cases': total,
        'skipped_cases': len(skipped),
        'passed_cases': passed,
        'failed_cases': total - passed,
        'recall_at_1': sum(1 for result in evaluated if result['recall_at_1_hit']) / total,
        'recall_at_3': sum(1 for result in evaluated if result['recall_at_3_hit']) / total,
        'mrr': sum(result['reciprocal_rank'] for result in evaluated) / total,
        'exact_term_hit_rate': (
            0 if expected_term_total == 0 else term_hit_total / expected_term_total
        ),
        'average_latency_ms': (
            sum(result['latency_ms'] for result in evaluated) / total
        ),
    }


def run_evaluation(
    method_name: str = METHOD_NAME,
    write_output: bool = True,
    output_path: Path | None = None,
) -> dict:
    if method_name not in METHODS:
        raise ValueError(f'Unknown retrieval method: {method_name}')

    method_config = METHODS[method_name]
    selector = method_config['selector']
    cases = load_cases()
    per_case_results = [evaluate_case(case, selector) for case in cases]
    metrics = summarize(per_case_results)

    report = {
        'method_name': method_name,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': metrics,
        'per_case_results': per_case_results,
        'notes': method_config['notes'],
        'known_limitations': method_config['known_limitations'],
    }

    if write_output:
        output_path = output_path or method_config['output_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    return report


def run_all_evaluations(write_output: bool = True) -> list[dict]:
    return [
        run_evaluation(method_name=method_name, write_output=write_output)
        for method_name in METHODS
    ]


def print_summary(report: dict) -> None:
    metrics = report['metrics']
    output_path = METHODS[report['method_name']]['output_path']
    print(f"Method: {report['method_name']}")
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
    print(f"Report: {output_path.relative_to(REPO_ROOT)}")


def main() -> int:
    reports = run_all_evaluations(write_output=True)
    for index, report in enumerate(reports):
        if index:
            print()
        print_summary(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
