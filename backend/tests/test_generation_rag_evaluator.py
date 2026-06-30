"""Tests for the Phase 3 RAG speech-generation evaluator harness."""
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts import evaluate_generation_rag as evaluator  # noqa: E402
from utils.embedding_retrieval import DENSE_RETRIEVAL_METHOD  # noqa: E402
from utils.graphrag_retrieval import GRAPHRAG_RETRIEVAL_METHOD  # noqa: E402
from utils.lightrag_retrieval import LIGHTRAG_RETRIEVAL_METHOD  # noqa: E402


REQUIRED_CASE_FIELDS = {
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


def test_phase3_fixture_schema_and_case_count():
    cases = evaluator.load_generation_cases()

    assert len(cases) == 10
    assert {case['case_id'] for case in cases} == {
        'single_doc_exact_climate_en',
        'single_doc_paraphrase_health_fr',
        'single_doc_arabic_diplomacy_ar',
        'numbers_entities_precision_en',
        'multi_doc_direct_relation_food_health',
        'multi_doc_chain_climate_migration_health',
        'broad_synthesis_humanitarian_speech',
        'distractor_documents_regional_funding',
        'cross_language_request_fr_sources_en',
        'arabic_output_from_multidoc_sources',
    }
    for case in cases:
        assert REQUIRED_CASE_FIELDS.issubset(case)
        evaluator.validate_case_schema(case)
        assert case['source_documents']
        assert case['expected_source_documents']
        assert case['target_language'] in {'en', 'fr', 'ar'}


def test_each_method_produces_required_output_shape():
    case = evaluator.load_generation_cases()[0]

    for method_name in (
        DENSE_RETRIEVAL_METHOD,
        LIGHTRAG_RETRIEVAL_METHOD,
        GRAPHRAG_RETRIEVAL_METHOD,
    ):
        result = evaluator.evaluate_case_method(case, method_name)

        assert {
            'case_id',
            'retrieval_method',
            'target_language',
            'retrieved_evidence',
            'canonical_prompt',
            'generated_speech',
            'generation_mode',
            'context_character_count',
            'retrieved_source_documents',
            'retrieval_latency_ms',
            'generation_latency_ms',
            'grounding_proxy',
            'traceability_proxy',
        }.issubset(result)
        assert result['case_id'] == case['case_id']
        assert result['retrieval_method'] == method_name
        assert result['generation_mode'] == evaluator.GENERATION_MODE_MOCK
        assert result['retrieved_evidence']
        assert result['generated_speech'].startswith(f"[MOCK {case['target_language']} SPEECH]")


def test_canonical_prompt_controls_are_equal_except_evidence():
    case = evaluator.load_generation_cases()[6]
    results = [
        evaluator.evaluate_case_method(case, method_name)
        for method_name in (
            DENSE_RETRIEVAL_METHOD,
            LIGHTRAG_RETRIEVAL_METHOD,
            GRAPHRAG_RETRIEVAL_METHOD,
        )
    ]

    prompt_skeletons = {
        evaluator.canonical_prompt_without_evidence(result['canonical_prompt'])
        for result in results
    }

    assert len(prompt_skeletons) == 1
    for result in results:
        controls = result['prompt_controls']
        assert controls['target_language'] == case['target_language']
        assert controls['requested_duration_seconds'] == case['requested_duration_seconds']
        assert controls['user_request'] == case['user_request']
        assert 'retrieval method' in result['canonical_prompt'].lower()


def test_all_methods_use_same_context_budget():
    case = evaluator.load_generation_cases()[4]
    results = [
        evaluator.evaluate_case_method(case, method_name)
        for method_name in (
            DENSE_RETRIEVAL_METHOD,
            LIGHTRAG_RETRIEVAL_METHOD,
            GRAPHRAG_RETRIEVAL_METHOD,
        )
    ]

    assert {result['max_context_character_budget'] for result in results} == {
        evaluator.MAX_EVIDENCE_CHARACTERS
    }
    assert {result['max_evidence_chunks'] for result in results} == {
        evaluator.MAX_EVIDENCE_CHUNKS
    }
    assert all(result['grounding_proxy']['within_context_budget'] for result in results)


def test_mock_generator_is_deterministic():
    case = evaluator.load_generation_cases()[0]
    evidence = evaluator.deterministic_dense_stub_selector(
        evaluator.load_case_chunk_records(case),
        {'query': case['user_request'], 'language': case['target_language']},
        max_excerpts=evaluator.MAX_EVIDENCE_CHUNKS,
        max_total_characters=evaluator.MAX_EVIDENCE_CHARACTERS,
    )
    prompt = evaluator.build_canonical_prompt(case, evidence)

    first = evaluator.mock_generation_function(prompt, case, evidence)
    second = evaluator.mock_generation_function(prompt, case, evidence)

    assert first == second
    assert 'pipeline only' in first


def test_no_production_module_a_import_is_required():
    source = Path(evaluator.__file__).read_text(encoding='utf-8')

    assert 'modules.module_a' not in source
    assert "from modules import module_a" not in source
    assert 'from flask' not in source.lower()
    assert 'import flask' not in source.lower()


def test_graph_helpers_are_evaluator_only_imports():
    source = Path(evaluator.__file__).read_text(encoding='utf-8')

    assert 'select_relevant_chunks_lightrag_with_metadata' in source
    assert 'select_relevant_chunks_graphrag_with_metadata' in source
    assert 'module_a_bp' not in source


def test_run_generation_evaluation_all_cases_mock_mode():
    report = evaluator.run_generation_evaluation()

    assert report['benchmark_name'] == 'phase3_end_to_end_speech_generation_rag'
    assert report['generation_mode'] == evaluator.GENERATION_MODE_MOCK
    assert report['case_count'] == 10
    assert report['result_count'] == 30
    assert report['real_llm_called'] is False
    assert set(report['summary']) == {
        DENSE_RETRIEVAL_METHOD,
        LIGHTRAG_RETRIEVAL_METHOD,
        GRAPHRAG_RETRIEVAL_METHOD,
    }
    for metrics in report['summary'].values():
        assert metrics['all_within_context_budget'] is True
