"""
Tests for the retrieval evaluator output shape and baseline metrics.
"""
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.evaluate_retrieval import run_evaluation  # noqa: E402


def test_retrieval_evaluator_metrics_shape():
    report = run_evaluation(write_output=False)
    metrics = report['metrics']

    assert report['method_name'] == 'rag_lite_keyword_metadata'
    assert metrics['total_cases'] > 0
    assert metrics['evaluated_cases'] > 0
    assert metrics['recall_at_1'] >= 0.5
    assert metrics['recall_at_3'] >= metrics['recall_at_1']
    assert metrics['mrr'] >= 0
    assert 'per_case_results' in report
    assert report['per_case_results']
