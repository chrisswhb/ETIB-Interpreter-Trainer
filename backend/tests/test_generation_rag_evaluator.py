"""Tests for the Phase 3 RAG speech-generation evaluator harness."""
from pathlib import Path
import sys

import pytest


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
            'requested_duration_seconds',
            'retrieved_evidence',
            'canonical_prompt',
            'generated_speech',
            'generation_error',
            'generation_mode',
            'provider',
            'model',
            'temperature',
            'max_tokens',
            'prompt_template_hash',
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
        assert result['provider'] is None
        assert result['model'] is None
        assert result['generation_error'] is None
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
    assert len({result['prompt_template_hash'] for result in results}) == 1


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


def test_mock_mode_remains_default():
    report = evaluator.run_generation_evaluation(method_names=[LIGHTRAG_RETRIEVAL_METHOD])

    assert report['generation_mode'] == evaluator.GENERATION_MODE_MOCK
    assert report['real_llm_called'] is False
    assert report['provider'] is None
    assert report['model'] is None


def test_repository_root_resolution_is_independent_of_current_working_directory(monkeypatch, tmp_path):
    repo_root = tmp_path / 'repo'
    script_path = repo_root / 'backend' / 'scripts' / 'evaluate_generation_rag.py'
    script_path.parent.mkdir(parents=True)
    script_path.write_text('# placeholder', encoding='utf-8')
    unrelated_dir = tmp_path / 'elsewhere'
    unrelated_dir.mkdir()

    monkeypatch.chdir(unrelated_dir)

    assert evaluator.resolve_repository_root(script_path) == repo_root


def test_load_evaluator_dotenv_uses_root_env_without_overriding_shell(monkeypatch, tmp_path):
    repo_root = tmp_path / 'repo'
    repo_root.mkdir()
    dotenv_path = repo_root / '.env'
    dotenv_path.write_text('GOOGLE_AI_KEY=from_dotenv_for_test_only\n', encoding='utf-8')
    calls = []

    def fake_load_dotenv(path, override):
        calls.append((path, override))
        if override is False and not evaluator.os.getenv('GOOGLE_AI_KEY'):
            monkeypatch.setenv('GOOGLE_AI_KEY', 'from_dotenv_for_test_only')

    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)
    monkeypatch.setattr(evaluator, 'load_dotenv', fake_load_dotenv)

    assert evaluator.load_evaluator_dotenv(repo_root) is True
    assert calls == [(dotenv_path, False)]
    assert evaluator.os.getenv('GOOGLE_AI_KEY') == 'from_dotenv_for_test_only'

    monkeypatch.setenv('GOOGLE_AI_KEY', 'from_shell_for_test_only')
    evaluator.load_evaluator_dotenv(repo_root)
    assert evaluator.os.getenv('GOOGLE_AI_KEY') == 'from_shell_for_test_only'


def test_preflight_attempts_dotenv_before_provider_validation(monkeypatch):
    calls = []

    def fake_load_evaluator_dotenv():
        calls.append('loaded')
        monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
        return True

    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)
    monkeypatch.setattr(evaluator, 'load_evaluator_dotenv', fake_load_evaluator_dotenv)

    preflight = evaluator.preflight_real_generation(
        method_names=[LIGHTRAG_RETRIEVAL_METHOD],
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=evaluator.BACKEND_ROOT / 'reports' / 'rag_results' / 'test_preflight.json',
    )

    assert calls == ['loaded']
    assert preflight['expected_generation_count'] == 10
    assert preflight['llm_called'] is False


def test_mock_mode_does_not_require_google_key(monkeypatch):
    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)

    report = evaluator.run_generation_evaluation(method_names=[GRAPHRAG_RETRIEVAL_METHOD])

    assert report['generation_mode'] == evaluator.GENERATION_MODE_MOCK
    assert report['result_count'] == 10
    assert report['real_llm_called'] is False


def test_preflight_performs_no_provider_call(monkeypatch, tmp_path):
    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')

    def fail_if_called(*args, **kwargs):
        raise AssertionError('generate_text should not be called during preflight')

    from services import llm_service

    monkeypatch.setattr(llm_service, 'generate_text', fail_if_called)
    output_path = evaluator.BACKEND_ROOT / 'reports' / 'rag_results' / 'test_preflight.json'

    preflight = evaluator.preflight_real_generation(
        method_names=None,
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=output_path,
    )

    assert preflight['provider'] == 'gemini'
    assert preflight['model'] == 'gemini-1.5-flash-latest'
    assert preflight['expected_generation_count'] == 30
    assert preflight['llm_called'] is False


def test_real_mode_refuses_to_run_when_gemini_key_absent(monkeypatch):
    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.validate_generation_controls(
            evaluator.GENERATION_MODE_REAL,
            'gemini',
            temperature=0,
            max_tokens=2800,
        )

    assert 'GOOGLE_AI_KEY' in str(exc_info.value)


def test_real_mode_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.validate_generation_controls(
            evaluator.GENERATION_MODE_REAL,
            'unsupported_provider',
            temperature=0,
            max_tokens=2800,
        )

    assert 'Unsupported real-generation provider' in str(exc_info.value)


def test_real_mode_preserves_same_generation_controls_across_methods():
    case = evaluator.load_generation_cases()[2]

    def fake_real_generation(prompt, case, evidence):
        return f"REAL {case['target_language']} {len(evidence)}"

    results = [
        evaluator.evaluate_case_method(
            case,
            method_name,
            generation_function=fake_real_generation,
            generation_mode=evaluator.GENERATION_MODE_REAL,
            provider='gemini',
            model='gemini-1.5-flash-latest',
            temperature=0,
            max_tokens=2800,
        )
        for method_name in (
            DENSE_RETRIEVAL_METHOD,
            LIGHTRAG_RETRIEVAL_METHOD,
            GRAPHRAG_RETRIEVAL_METHOD,
        )
    ]

    assert {result['provider'] for result in results} == {'gemini'}
    assert {result['model'] for result in results} == {'gemini-1.5-flash-latest'}
    assert {result['temperature'] for result in results} == {0}
    assert {result['max_tokens'] for result in results} == {2800}
    assert {result['max_evidence_chunks'] for result in results} == {evaluator.MAX_EVIDENCE_CHUNKS}
    assert {result['max_context_character_budget'] for result in results} == {
        evaluator.MAX_EVIDENCE_CHARACTERS
    }


def test_retrieved_evidence_can_differ_while_shared_template_hash_matches():
    case = evaluator.load_generation_cases()[7]
    results = [
        evaluator.evaluate_case_method(case, method_name)
        for method_name in (
            DENSE_RETRIEVAL_METHOD,
            LIGHTRAG_RETRIEVAL_METHOD,
            GRAPHRAG_RETRIEVAL_METHOD,
        )
    ]

    evidence_sets = {
        tuple(result['retrieved_source_documents'])
        for result in results
    }

    assert len({result['prompt_template_hash'] for result in results}) == 1
    assert len(evidence_sets) >= 1
    assert {
        evaluator.canonical_prompt_without_evidence(result['canonical_prompt'])
        for result in results
    } == {results[0]['prompt_controls']['template_without_evidence']}


def test_real_result_schema_captures_generation_errors_without_secret_values():
    case = evaluator.load_generation_cases()[0]

    def failing_generation(prompt, case, evidence):
        raise RuntimeError('GOOGLE_AI_KEY is not configured')

    result = evaluator.evaluate_case_method(
        case,
        LIGHTRAG_RETRIEVAL_METHOD,
        generation_function=failing_generation,
        generation_mode=evaluator.GENERATION_MODE_REAL,
        provider='gemini',
        model='gemini-1.5-flash-latest',
        temperature=0,
        max_tokens=2800,
    )

    assert result['generated_speech'] == ''
    assert result['generation_error'] == 'GOOGLE_AI_KEY is not configured'
    assert result['provider'] == 'gemini'
    assert result['prompt_template_hash']


def test_real_generation_wrapper_calls_service_directly_and_restores_state(monkeypatch):
    from services import llm_service

    original_provider = llm_service.LLM_PROVIDER
    original_model = llm_service.GEMINI_MODEL
    captured = {}

    def fake_generate_text(messages, max_tokens, temperature):
        captured['messages'] = messages
        captured['max_tokens'] = max_tokens
        captured['temperature'] = temperature
        captured['provider_during_call'] = llm_service.LLM_PROVIDER
        captured['model_during_call'] = llm_service.GEMINI_MODEL
        return 'generated text'

    monkeypatch.setattr(llm_service, 'generate_text', fake_generate_text)
    generator = evaluator.real_generation_function_factory(
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        model_override='gemini-test-model',
    )

    output = generator('Prompt body', {'target_language': 'en'}, [])

    assert output == 'generated text'
    assert captured['provider_during_call'] == 'gemini'
    assert captured['model_during_call'] == 'gemini-test-model'
    assert captured['max_tokens'] == 2800
    assert captured['temperature'] == 0
    assert captured['messages'][0]['role'] == 'system'
    assert captured['messages'][1] == {'role': 'user', 'content': 'Prompt body'}
    assert llm_service.LLM_PROVIDER == original_provider
    assert llm_service.GEMINI_MODEL == original_model
