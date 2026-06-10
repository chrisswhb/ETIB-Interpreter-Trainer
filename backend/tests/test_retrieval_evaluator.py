"""
Tests for the retrieval evaluator output shape and baseline metrics.
"""
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.evaluate_retrieval import (  # noqa: E402
    BM25_METHOD_NAME,
    DENSE_METHOD_NAME,
    METHOD_NAME,
    load_cases,
    run_all_evaluations,
    run_evaluation,
)


def test_retrieval_evaluator_metrics_shape():
    report = run_evaluation(write_output=False)
    metrics = report['metrics']

    assert report['method_name'] == METHOD_NAME
    assert metrics['total_cases'] > 0
    assert metrics['evaluated_cases'] > 0
    assert metrics['recall_at_1'] >= 0.5
    assert metrics['recall_at_3'] >= metrics['recall_at_1']
    assert metrics['mrr'] >= 0
    assert 'metrics_by_category' in metrics
    assert 'metrics_by_difficulty' in metrics
    assert 'per_method_strengths_and_weaknesses' in report
    assert 'per_case_results' in report
    assert report['per_case_results']


def test_retrieval_evaluator_runs_all_methods():
    reports = run_all_evaluations(write_output=False)
    by_method = {report['method_name']: report for report in reports}

    assert set(by_method) == {METHOD_NAME, BM25_METHOD_NAME, DENSE_METHOD_NAME}
    for method_name in (METHOD_NAME, BM25_METHOD_NAME):
        report = by_method[method_name]
        metrics = report['metrics']
        assert report['method_available'] is True
        assert metrics['total_cases'] > 0
        assert metrics['evaluated_cases'] > 0
        assert metrics['recall_at_3'] >= metrics['recall_at_1']
        assert 'per_case_results' in report
        assert report['per_case_results']

    dense_report = by_method[DENSE_METHOD_NAME]
    assert 'method_available' in dense_report
    assert dense_report['metrics']['total_cases'] > 0
    assert 'per_case_results' in dense_report
    assert dense_report['per_case_results']


def test_retrieval_cases_include_harder_and_future_baselines():
    cases = load_cases()
    categories = {case['category'] for case in cases}
    evaluated_categories = {
        result['category']
        for result in run_evaluation(write_output=False)['per_case_results']
        if not result.get('skipped')
    }

    assert len(cases) >= 12
    assert all('difficulty' in case for case in cases)
    assert all('expected_best_methods' in case for case in cases)
    assert all('expected_weak_methods' in case for case in cases)
    assert 'noisy_ranking' in categories
    assert 'rare_terminology' in categories
    assert 'semantic_paraphrase' in evaluated_categories
    assert 'cross_language' in evaluated_categories
    assert 'graph_relation_placeholder' in evaluated_categories


def test_dense_evaluator_skips_cleanly_when_unavailable(monkeypatch):
    import scripts.evaluate_retrieval as evaluator

    monkeypatch.setitem(
        evaluator.METHODS[DENSE_METHOD_NAME],
        'availability_check',
        lambda: False,
    )

    report = run_evaluation(method_name=DENSE_METHOD_NAME, write_output=False)
    metrics = report['metrics']

    assert report['method_name'] == DENSE_METHOD_NAME
    assert report['method_available'] is False
    assert 'sentence-transformers' in report['unavailable_reason']
    assert 'backend/requirements-rag-optional.txt' in report['unavailable_reason']
    assert metrics['total_cases'] > 0
    assert metrics['evaluated_cases'] == 0
    assert metrics['skipped_cases'] == metrics['total_cases']


def test_dense_evaluator_output_shape_when_selector_is_available(monkeypatch):
    import scripts.evaluate_retrieval as evaluator

    def fake_dense_selector(records, params, max_excerpts, max_total_characters):
        selected = []
        for record in records[:max_excerpts]:
            selected.append({
                'text': record['text'],
                'source_filename': record['source_filename'],
                'source_type': record['source_type'],
                'chunk_index': record['chunk_index'],
                'score': 0.75,
                'retrieval_method': DENSE_METHOD_NAME,
            })
        return selected

    monkeypatch.setitem(
        evaluator.METHODS[DENSE_METHOD_NAME],
        'availability_check',
        lambda: True,
    )
    monkeypatch.setitem(
        evaluator.METHODS[DENSE_METHOD_NAME],
        'selector',
        fake_dense_selector,
    )

    report = run_evaluation(method_name=DENSE_METHOD_NAME, write_output=False)
    first_selected = report['per_case_results'][0]['selected_chunks'][0]

    assert report['method_available'] is True
    assert report['metrics']['evaluated_cases'] > 0
    assert set(first_selected) == {
        'text',
        'source_filename',
        'source_type',
        'chunk_index',
        'score',
        'retrieval_method',
    }
    assert first_selected['retrieval_method'] == DENSE_METHOD_NAME
