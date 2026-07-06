"""Tests for the Phase 3 RAG speech-generation evaluator harness."""
import json
from pathlib import Path
import sys

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts import evaluate_generation_rag as evaluator  # noqa: E402
from utils.embedding_retrieval import DENSE_RETRIEVAL_METHOD, DenseEmbeddingUnavailable  # noqa: E402
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

ORIGINAL_PHASE3_CASE_SOURCES = {
    'single_doc_exact_climate_en': {
        'source_documents': ['climate_finance_en.txt'],
        'expected_source_documents': ['climate_finance_en.txt'],
    },
    'single_doc_paraphrase_health_fr': {
        'source_documents': ['health_systems_fr.txt'],
        'expected_source_documents': ['health_systems_fr.txt'],
    },
    'single_doc_arabic_diplomacy_ar': {
        'source_documents': ['arab_diplomacy_ar.txt'],
        'expected_source_documents': ['arab_diplomacy_ar.txt'],
    },
    'numbers_entities_precision_en': {
        'source_documents': ['numbers_entities_en.txt'],
        'expected_source_documents': ['numbers_entities_en.txt'],
    },
    'multi_doc_direct_relation_food_health': {
        'source_documents': ['food_security_migration_en.txt', 'health_displacement_en.txt'],
        'expected_source_documents': ['food_security_migration_en.txt', 'health_displacement_en.txt'],
    },
    'multi_doc_chain_climate_migration_health': {
        'source_documents': [
            'climate_agriculture_en.txt',
            'food_security_migration_en.txt',
            'health_displacement_en.txt',
        ],
        'expected_source_documents': [
            'climate_agriculture_en.txt',
            'food_security_migration_en.txt',
            'health_displacement_en.txt',
        ],
    },
    'broad_synthesis_humanitarian_speech': {
        'source_documents': [
            'climate_agriculture_en.txt',
            'food_security_migration_en.txt',
            'health_displacement_en.txt',
            'regional_funding_en.txt',
            'cooperation_policy_en.txt',
        ],
        'expected_source_documents': [
            'climate_agriculture_en.txt',
            'food_security_migration_en.txt',
            'health_displacement_en.txt',
        ],
    },
    'distractor_documents_regional_funding': {
        'source_documents': [
            'regional_funding_en.txt',
            'distractor_sports_en.txt',
            'cooperation_policy_en.txt',
        ],
        'expected_source_documents': ['regional_funding_en.txt', 'cooperation_policy_en.txt'],
    },
    'cross_language_request_fr_sources_en': {
        'source_documents': ['climate_finance_en.txt', 'regional_funding_en.txt'],
        'expected_source_documents': ['climate_finance_en.txt', 'regional_funding_en.txt'],
    },
    'arabic_output_from_multidoc_sources': {
        'source_documents': [
            'food_security_migration_en.txt',
            'health_systems_fr.txt',
            'regional_funding_en.txt',
        ],
        'expected_source_documents': [
            'food_security_migration_en.txt',
            'health_systems_fr.txt',
            'regional_funding_en.txt',
        ],
    },
}

HARD_PHASE3_CASE_IDS = {
    'multi_doc_chain_climate_migration_health_hard',
    'cross_language_request_fr_sources_en_hard',
    'arabic_output_from_multidoc_sources_hard',
}

PHASE3_CASE_COUNT = len(ORIGINAL_PHASE3_CASE_SOURCES) + len(HARD_PHASE3_CASE_IDS)


def test_phase3_fixture_schema_and_case_count():
    cases = evaluator.load_generation_cases()
    case_ids = {case['case_id'] for case in cases}

    assert len(cases) == PHASE3_CASE_COUNT
    assert case_ids == set(ORIGINAL_PHASE3_CASE_SOURCES) | HARD_PHASE3_CASE_IDS
    for case in cases:
        assert REQUIRED_CASE_FIELDS.issubset(case)
        evaluator.validate_case_schema(case)
        assert case['source_documents']
        assert case['expected_source_documents']
        assert case['target_language'] in {'en', 'fr', 'ar'}


def test_original_phase3_cases_remain_unchanged():
    cases_by_id = {case['case_id']: case for case in evaluator.load_generation_cases()}

    for case_id, expected in ORIGINAL_PHASE3_CASE_SOURCES.items():
        case = cases_by_id[case_id]
        assert case['source_documents'] == expected['source_documents']
        assert case['expected_source_documents'] == expected['expected_source_documents']


def test_hard_phase3_cases_force_three_chunk_tradeoffs():
    cases_by_id = {case['case_id']: case for case in evaluator.load_generation_cases()}

    for case_id in HARD_PHASE3_CASE_IDS:
        case = cases_by_id[case_id]
        assert len(case['source_documents']) > evaluator.MAX_EVIDENCE_CHUNKS
        assert set(case['expected_source_documents']).issubset(case['source_documents'])
        assert set(case['distractor_documents']).issubset(case['source_documents'])
        assert set(case['distractor_documents']).isdisjoint(case['expected_source_documents'])
        assert case['distractor_rationale']
        assert case['hard_case_design_notes']


def test_select_generation_cases_default_returns_all_cases():
    cases = evaluator.load_generation_cases()

    selected = evaluator.select_generation_cases(cases)

    assert len(selected) == PHASE3_CASE_COUNT
    assert selected == cases


def test_select_generation_cases_valid_case_id_returns_one_case():
    cases = evaluator.load_generation_cases()

    selected = evaluator.select_generation_cases(cases, 'single_doc_exact_climate_en')

    assert len(selected) == 1
    assert selected[0]['case_id'] == 'single_doc_exact_climate_en'


def test_each_hard_phase3_case_can_be_selected_for_mock_evaluation():
    for case_id in HARD_PHASE3_CASE_IDS:
        report = evaluator.run_generation_evaluation(
            method_names=[DENSE_RETRIEVAL_METHOD],
            case_id=case_id,
            dense_mode=evaluator.DENSE_MODE_STUB,
        )

        assert report['generation_mode'] == evaluator.GENERATION_MODE_MOCK
        assert report['real_llm_called'] is False
        assert report['case_count'] == 1
        assert report['result_count'] == 1
        assert report['results'][0]['case_id'] == case_id
        assert report['results'][0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD


def test_select_generation_cases_invalid_case_id_fails_safely():
    cases = evaluator.load_generation_cases()

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.select_generation_cases(cases, 'not_a_real_case')

    message = str(exc_info.value)
    assert "Unknown case_id 'not_a_real_case'" in message
    assert 'single_doc_exact_climate_en' in message
    assert 'climate mitigation measures' not in message


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
            'dense_mode',
            'provider',
            'model',
            'temperature',
            'max_tokens',
            'thinking_budget_requested',
            'prompt_template_hash',
            'context_character_count',
            'retrieved_source_documents',
            'retrieval_latency_ms',
            'generation_latency_ms',
            'provider_finish_reason',
            'provider_usage_metadata',
            'provider_safety_metadata',
            'provider_candidate_count',
            'provider_response_metadata_available',
            'provider_thoughts_token_count',
            'grounding_proxy',
            'traceability_proxy',
        }.issubset(result)
        assert result['case_id'] == case['case_id']
        assert result['retrieval_method'] == method_name
        assert result['generation_mode'] == evaluator.GENERATION_MODE_MOCK
        assert result['dense_mode'] == evaluator.DENSE_MODE_STUB
        assert result['provider'] is None
        assert result['model'] is None
        assert result['provider_finish_reason'] is None
        assert result['provider_usage_metadata'] is None
        assert result['provider_safety_metadata'] is None
        assert result['provider_candidate_count'] is None
        assert result['provider_response_metadata_available'] is False
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
    assert report['dense_mode'] == evaluator.DENSE_MODE_STUB
    assert report['case_count'] == PHASE3_CASE_COUNT
    assert report['result_count'] == PHASE3_CASE_COUNT * 3
    assert report['real_llm_called'] is False
    assert set(report['summary']) == {
        DENSE_RETRIEVAL_METHOD,
        LIGHTRAG_RETRIEVAL_METHOD,
        GRAPHRAG_RETRIEVAL_METHOD,
    }
    for metrics in report['summary'].values():
        assert metrics['all_within_context_budget'] is True


def test_run_generation_evaluation_case_id_and_method_produces_one_result():
    report = evaluator.run_generation_evaluation(
        method_names=[DENSE_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
    )

    assert report['case_count'] == 1
    assert report['result_count'] == 1
    assert report['case_id_filter'] == 'single_doc_exact_climate_en'
    assert report['method_names'] == [DENSE_RETRIEVAL_METHOD]
    assert report['results'][0]['case_id'] == 'single_doc_exact_climate_en'
    assert report['results'][0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD


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
        case_id=None,
        dense_mode=evaluator.DENSE_MODE_STUB,
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=evaluator.BACKEND_ROOT / 'reports' / 'rag_results' / 'test_preflight.json',
    )

    assert calls == ['loaded']
    assert preflight['expected_generation_count'] == PHASE3_CASE_COUNT
    assert preflight['llm_called'] is False


def test_preflight_with_case_id_and_method_reports_one_generation(monkeypatch):
    def fake_load_evaluator_dotenv():
        monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
        return True

    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)
    monkeypatch.setattr(evaluator, 'load_evaluator_dotenv', fake_load_evaluator_dotenv)

    preflight = evaluator.preflight_real_generation(
        method_names=[DENSE_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
        dense_mode=evaluator.DENSE_MODE_STUB,
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=evaluator.BACKEND_ROOT / 'reports' / 'rag_results' / 'test_preflight.json',
    )

    assert preflight['case_count'] == 1
    assert preflight['method_count'] == 1
    assert preflight['expected_generation_count'] == 1
    assert preflight['thinking_budget'] is None
    assert preflight['llm_called'] is False


def test_preflight_reports_requested_thinking_budget(monkeypatch):
    def fake_load_evaluator_dotenv():
        monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
        return True

    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)
    monkeypatch.setattr(evaluator, 'load_evaluator_dotenv', fake_load_evaluator_dotenv)

    preflight = evaluator.preflight_real_generation(
        method_names=[DENSE_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
        dense_mode=evaluator.DENSE_MODE_STUB,
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=evaluator.BACKEND_ROOT / 'reports' / 'rag_results' / 'test_preflight.json',
        thinking_budget=256,
    )

    assert preflight['thinking_budget'] == 256
    assert preflight['expected_generation_count'] == 1


def test_mock_mode_does_not_require_google_key(monkeypatch):
    monkeypatch.delenv('GOOGLE_AI_KEY', raising=False)

    report = evaluator.run_generation_evaluation(method_names=[GRAPHRAG_RETRIEVAL_METHOD])

    assert report['generation_mode'] == evaluator.GENERATION_MODE_MOCK
    assert report['result_count'] == PHASE3_CASE_COUNT
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
        case_id=None,
        dense_mode=evaluator.DENSE_MODE_STUB,
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        output_path=output_path,
    )

    assert preflight['provider'] == 'gemini'
    assert preflight['model'] == 'gemini-1.5-flash-latest'
    assert preflight['expected_generation_count'] == PHASE3_CASE_COUNT * 3
    assert preflight['llm_called'] is False


def test_report_output_path_display_handles_relative_and_absolute_paths():
    relative_path = Path('backend/reports/rag_results/path_regression.json')
    absolute_path = evaluator.REPO_ROOT / relative_path

    assert evaluator.display_report_output_path(relative_path) == str(relative_path)
    assert evaluator.display_report_output_path(absolute_path) == str(relative_path)


def test_validate_relative_output_path_is_independent_of_cwd(monkeypatch, tmp_path):
    unrelated_dir = tmp_path / 'elsewhere'
    unrelated_dir.mkdir()
    monkeypatch.chdir(unrelated_dir)

    evaluator.validate_output_path(Path('backend/reports/rag_results/path_regression.json'))


def test_mock_write_reports_output_path_without_crashing(monkeypatch, capsys):
    output_path = Path('backend/reports/rag_results/phase3_mock_write_path_regression_pytest.json')
    resolved_output = evaluator.resolve_report_output_path(output_path)
    if resolved_output.exists():
        resolved_output.unlink()

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'evaluate_generation_rag.py',
            '--generation-mode',
            'mock',
            '--case-id',
            'single_doc_exact_climate_en',
            '--method',
            DENSE_RETRIEVAL_METHOD,
            '--output',
            str(output_path),
            '--write',
        ],
    )

    assert evaluator.main() == 0

    captured = capsys.readouterr().out
    assert 'Report: backend' in captured
    assert resolved_output.exists()
    report = json.loads(resolved_output.read_text(encoding='utf-8'))
    assert report['case_count'] == 1
    assert report['result_count'] == 1


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


def test_invalid_evaluator_thinking_budget_fails_before_generation(monkeypatch):
    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.validate_generation_controls(
            evaluator.GENERATION_MODE_REAL,
            'gemini',
            temperature=0,
            max_tokens=2800,
            thinking_budget=-1,
        )

    assert 'thinking_budget' in str(exc_info.value)


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
    assert {result['thinking_budget_requested'] for result in results} == {None}
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

    def fake_generate_text(
        messages,
        max_tokens,
        temperature,
        return_metadata=False,
        thinking_budget=None,
    ):
        captured['messages'] = messages
        captured['max_tokens'] = max_tokens
        captured['temperature'] = temperature
        captured['return_metadata'] = return_metadata
        captured['thinking_budget'] = thinking_budget
        captured['provider_during_call'] = llm_service.LLM_PROVIDER
        captured['model_during_call'] = llm_service.GEMINI_MODEL
        return {
            'text': 'generated text',
            'finish_reason': 'STOP',
            'usage_metadata': {'totalTokenCount': 12},
            'safety_metadata': [{'category': 'HARM_CATEGORY_TEST', 'probability': 'NEGLIGIBLE'}],
            'candidate_count': 1,
            'response_metadata_available': True,
        } if return_metadata else 'generated text'

    monkeypatch.setattr(llm_service, 'generate_text', fake_generate_text)
    generator = evaluator.real_generation_function_factory(
        provider='gemini',
        temperature=0,
        max_tokens=2800,
        model_override='gemini-test-model',
    )

    output = generator('Prompt body', {'target_language': 'en'}, [])

    assert isinstance(output, dict)
    assert captured['provider_during_call'] == 'gemini'
    assert captured['model_during_call'] == 'gemini-test-model'
    assert captured['max_tokens'] == 2800
    assert captured['temperature'] == 0
    assert captured['return_metadata'] is True
    assert captured['thinking_budget'] is None
    assert captured['messages'][0]['role'] == 'system'
    assert captured['messages'][1] == {'role': 'user', 'content': 'Prompt body'}
    assert output['text'] == 'generated text'
    assert output['finish_reason'] == 'STOP'
    assert llm_service.LLM_PROVIDER == original_provider
    assert llm_service.GEMINI_MODEL == original_model


def test_llm_service_default_generate_text_returns_plain_string(monkeypatch):
    from services import llm_service
    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200
        text = ''

        def json(self):
            return {
                'candidates': [
                    {
                        'content': {'parts': [{'text': ' plain text output '}]},
                        'finishReason': 'STOP',
                    }
                ],
                'usageMetadata': {'totalTokenCount': 17},
            }

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(llm_service, 'LLM_PROVIDER', 'gemini')
    monkeypatch.setattr(llm_service, 'GEMINI_MODEL', 'gemini-test-model')
    def fake_post(url, json, timeout):
        captured['generation_config'] = json['generationConfig']
        return FakeResponse()

    monkeypatch.setattr('requests.post', fake_post)

    output = llm_service.generate_text(
        [{'role': 'user', 'content': 'hello'}],
        max_tokens=42,
        temperature=0,
    )

    assert output == 'plain text output'
    assert captured['generation_config'] == {'maxOutputTokens': 42, 'temperature': 0}


def test_llm_service_gemini_return_metadata_normalizes_safe_fields(monkeypatch):
    from services import llm_service

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200
        text = ''

        def json(self):
            return {
                'candidates': [
                    {
                        'content': {'parts': [{'text': 'metadata text'}]},
                        'finishReason': 'MAX_TOKENS',
                        'safetyRatings': [
                            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'LOW'}
                        ],
                    },
                    {
                        'content': {'parts': [{'text': 'unused candidate'}]},
                        'finishReason': 'STOP',
                    },
                ],
                'usageMetadata': {
                    'promptTokenCount': 10,
                    'candidatesTokenCount': 5,
                    'totalTokenCount': 15,
                },
            }

    def fake_post(url, json, timeout):
        captured['json'] = json
        captured['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(llm_service, 'LLM_PROVIDER', 'gemini')
    monkeypatch.setattr(llm_service, 'GEMINI_MODEL', 'gemini-test-model')
    monkeypatch.setattr('requests.post', fake_post)

    output = llm_service.generate_text(
        [{'role': 'user', 'content': 'hello'}],
        max_tokens=42,
        temperature=0,
        thinking_budget=256,
        return_metadata=True,
    )

    assert output == {
        'text': 'metadata text',
        'provider': 'gemini',
        'model': 'gemini-test-model',
        'finish_reason': 'MAX_TOKENS',
        'usage_metadata': {
            'promptTokenCount': 10,
            'candidatesTokenCount': 5,
            'totalTokenCount': 15,
        },
        'safety_metadata': [
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'LOW'}
        ],
        'candidate_count': 2,
        'response_metadata_available': True,
    }
    assert captured['json']['generationConfig'] == {
        'maxOutputTokens': 42,
        'temperature': 0,
        'thinkingConfig': {'thinkingBudget': 256},
    }
    assert 'configured-for-test-only' not in str(output)


def test_invalid_thinking_budget_fails_before_provider_call(monkeypatch):
    from services import llm_service

    def fail_post(*args, **kwargs):
        raise AssertionError('provider call should not happen for invalid thinking_budget')

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(llm_service, 'LLM_PROVIDER', 'gemini')
    monkeypatch.setattr('requests.post', fail_post)

    with pytest.raises(ValueError):
        llm_service.generate_text(
            [{'role': 'user', 'content': 'hello'}],
            max_tokens=42,
            temperature=0,
            thinking_budget=0,
        )


def test_non_gemini_provider_ignores_valid_thinking_budget(monkeypatch):
    from services import llm_service

    captured = {}

    def fake_remote(messages, max_tokens, temperature):
        captured['messages'] = messages
        captured['max_tokens'] = max_tokens
        captured['temperature'] = temperature
        return 'remote output'

    monkeypatch.setattr(llm_service, 'LLM_PROVIDER', 'remote_aya')
    monkeypatch.setattr(llm_service, '_generate_with_remote_aya', fake_remote)

    output = llm_service.generate_text(
        [{'role': 'user', 'content': 'hello'}],
        max_tokens=42,
        temperature=0.2,
        thinking_budget=256,
    )

    assert output == 'remote output'
    assert captured == {
        'messages': [{'role': 'user', 'content': 'hello'}],
        'max_tokens': 42,
        'temperature': 0.2,
    }


def test_evaluator_real_result_stores_provider_metadata():
    case = evaluator.load_generation_cases()[0]

    def metadata_generation(prompt, case, evidence):
        return {
            'text': 'real speech',
            'finish_reason': 'STOP',
            'usage_metadata': {'totalTokenCount': 99, 'thoughtsTokenCount': 55},
            'safety_metadata': [{'category': 'HARM_CATEGORY_TEST', 'probability': 'NEGLIGIBLE'}],
            'candidate_count': 1,
            'response_metadata_available': True,
        }

    result = evaluator.evaluate_case_method(
        case,
        LIGHTRAG_RETRIEVAL_METHOD,
        generation_function=metadata_generation,
        generation_mode=evaluator.GENERATION_MODE_REAL,
        provider='gemini',
        model='gemini-test-model',
        temperature=0,
        max_tokens=2800,
        thinking_budget=256,
    )

    assert result['generated_speech'] == 'real speech'
    assert result['thinking_budget_requested'] == 256
    assert result['provider_finish_reason'] == 'STOP'
    assert result['provider_usage_metadata'] == {'totalTokenCount': 99, 'thoughtsTokenCount': 55}
    assert result['provider_thoughts_token_count'] == 55
    assert result['provider_safety_metadata'] == [
        {'category': 'HARM_CATEGORY_TEST', 'probability': 'NEGLIGIBLE'}
    ]
    assert result['provider_candidate_count'] == 1
    assert result['provider_response_metadata_available'] is True


def test_missing_provider_metadata_does_not_crash_evaluation():
    case = evaluator.load_generation_cases()[0]

    result = evaluator.evaluate_case_method(
        case,
        LIGHTRAG_RETRIEVAL_METHOD,
        generation_function=lambda prompt, case, evidence: {'text': 'real speech'},
        generation_mode=evaluator.GENERATION_MODE_REAL,
        provider='gemini',
        model='gemini-test-model',
        temperature=0,
        max_tokens=2800,
    )

    assert result['generated_speech'] == 'real speech'
    assert result['provider_finish_reason'] is None
    assert result['provider_usage_metadata'] is None
    assert result['provider_safety_metadata'] is None
    assert result['provider_candidate_count'] is None
    assert result['provider_response_metadata_available'] is False


def test_gemini_model_listing_filters_text_generation_models(monkeypatch):
    class FakeResponse:
        ok = True
        status_code = 200
        text = ''

        def json(self):
            return {
                'models': [
                    {
                        'name': 'models/gemini-2.0-flash',
                        'supportedGenerationMethods': ['generateContent', 'countTokens'],
                    },
                    {
                        'name': 'models/embedding-001',
                        'supportedGenerationMethods': ['embedContent'],
                    },
                    {
                        'name': 'models/gemini-1.5-flash',
                        'supportedGenerationMethods': ['generateContent'],
                    },
                ]
            }

    captured = {}

    def fake_get(url, params, timeout):
        captured['called'] = True
        captured['has_key_param'] = 'key' in params
        captured['timeout'] = timeout
        return FakeResponse()

    import requests

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(requests, 'get', fake_get)

    models = evaluator.list_available_text_generation_models('gemini')

    assert captured == {'called': True, 'has_key_param': True, 'timeout': 60}
    assert models == ['gemini-1.5-flash', 'gemini-2.0-flash']


def test_list_available_models_cli_does_not_invoke_generation(monkeypatch, capsys):
    def fail_generation_factory(*args, **kwargs):
        raise AssertionError('generation should not be configured for listing')

    monkeypatch.setattr(
        evaluator,
        'list_available_text_generation_models',
        lambda provider: ['gemini-2.0-flash'],
    )
    monkeypatch.setattr(evaluator, 'real_generation_function_factory', fail_generation_factory)
    monkeypatch.setattr(
        sys,
        'argv',
        ['evaluate_generation_rag.py', '--provider', 'gemini', '--list-available-models'],
    )

    assert evaluator.main() == 0

    output = capsys.readouterr().out
    assert 'Available text-generation models:' in output
    assert 'gemini-2.0-flash' in output


def test_model_override_is_validated_and_carried_to_real_generation(monkeypatch):
    captured = {}

    def fake_factory(provider, temperature, max_tokens, model_override=None, thinking_budget=None):
        captured['provider'] = provider
        captured['temperature'] = temperature
        captured['max_tokens'] = max_tokens
        captured['model_override'] = model_override
        captured['thinking_budget'] = thinking_budget
        return lambda prompt, case, evidence: 'real output'

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(
        evaluator,
        'list_available_text_generation_models',
        lambda provider: ['gemini-2.0-flash'],
    )
    monkeypatch.setattr(evaluator, 'real_generation_function_factory', fake_factory)

    report = evaluator.run_generation_evaluation(
        method_names=[LIGHTRAG_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
        generation_mode=evaluator.GENERATION_MODE_REAL,
        provider='gemini',
        model='models/gemini-2.0-flash',
        temperature=0,
        max_tokens=2800,
    )

    assert captured == {
        'provider': 'gemini',
        'temperature': 0,
        'max_tokens': 2800,
        'model_override': 'models/gemini-2.0-flash',
        'thinking_budget': None,
    }
    assert report['model'] == 'gemini-2.0-flash'
    assert report['results'][0]['model'] == 'gemini-2.0-flash'
    assert report['results'][0]['generated_speech'] == 'real output'


def test_thinking_budget_is_forwarded_and_recorded_for_real_evaluation(monkeypatch):
    captured = {}

    def fake_factory(provider, temperature, max_tokens, model_override=None, thinking_budget=None):
        captured['thinking_budget'] = thinking_budget
        return lambda prompt, case, evidence: {
            'text': 'real output',
            'usage_metadata': {'thoughtsTokenCount': 12},
            'response_metadata_available': True,
        }

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(
        evaluator,
        'list_available_text_generation_models',
        lambda provider: ['gemini-2.0-flash'],
    )
    monkeypatch.setattr(evaluator, 'real_generation_function_factory', fake_factory)

    report = evaluator.run_generation_evaluation(
        method_names=[LIGHTRAG_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
        generation_mode=evaluator.GENERATION_MODE_REAL,
        provider='gemini',
        model='gemini-2.0-flash',
        temperature=0,
        max_tokens=2800,
        thinking_budget=256,
    )

    result = report['results'][0]
    assert captured['thinking_budget'] == 256
    assert report['thinking_budget_requested'] == 256
    assert result['thinking_budget_requested'] == 256
    assert result['provider_thoughts_token_count'] == 12


def test_unavailable_model_override_fails_before_generation(monkeypatch):
    def fail_generation_factory(*args, **kwargs):
        raise AssertionError('generation should not start for an unavailable model')

    monkeypatch.setenv('GOOGLE_AI_KEY', 'configured-for-test-only')
    monkeypatch.setattr(
        evaluator,
        'list_available_text_generation_models',
        lambda provider: ['gemini-2.0-flash'],
    )
    monkeypatch.setattr(evaluator, 'real_generation_function_factory', fail_generation_factory)

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.run_generation_evaluation(
            method_names=[LIGHTRAG_RETRIEVAL_METHOD],
            case_id='single_doc_exact_climate_en',
            generation_mode=evaluator.GENERATION_MODE_REAL,
            provider='gemini',
            model='gemini-missing-model',
            temperature=0,
            max_tokens=2800,
        )

    message = str(exc_info.value)
    assert "Gemini model 'gemini-missing-model' is not available" in message
    assert 'gemini-2.0-flash' in message
    assert 'configured-for-test-only' not in message


def test_default_dense_mode_remains_stub():
    case = evaluator.load_generation_cases()[0]

    evidence, _latency, runtime = evaluator.retrieve_evidence(
        case,
        DENSE_RETRIEVAL_METHOD,
    )

    assert runtime == 'deterministic_stub'
    assert evidence


def test_dense_mode_real_calls_real_dense_adapter(monkeypatch):
    case = evaluator.load_generation_cases()[0]
    calls = []

    def fake_dense_selector(records, params, max_excerpts, max_total_characters):
        calls.append({
            'record_count': len(records),
            'query': params['query'],
            'max_excerpts': max_excerpts,
            'max_total_characters': max_total_characters,
        })
        return [{
            'text': records[0]['text'],
            'source_filename': records[0]['source_filename'],
            'source_type': records[0]['source_type'],
            'chunk_index': records[0]['chunk_index'],
            'score': 0.99,
            'retrieval_method': DENSE_RETRIEVAL_METHOD,
        }]

    monkeypatch.setattr(evaluator, 'is_dense_embedding_available', lambda: True)
    monkeypatch.setattr(evaluator, 'select_relevant_chunks_dense_with_metadata', fake_dense_selector)

    evidence, _latency, runtime = evaluator.retrieve_evidence(
        case,
        DENSE_RETRIEVAL_METHOD,
        dense_mode=evaluator.DENSE_MODE_REAL,
    )

    assert runtime == 'real'
    assert calls == [{
        'record_count': 1,
        'query': case['user_request'],
        'max_excerpts': evaluator.MAX_EVIDENCE_CHUNKS,
        'max_total_characters': evaluator.MAX_EVIDENCE_CHARACTERS,
    }]
    assert evidence[0]['retrieval_method'] == DENSE_RETRIEVAL_METHOD


def test_dense_mode_real_fails_without_fallback_when_unavailable(monkeypatch):
    case = evaluator.load_generation_cases()[0]

    monkeypatch.setattr(evaluator, 'is_dense_embedding_available', lambda: True)

    def unavailable(*args, **kwargs):
        raise DenseEmbeddingUnavailable('model unavailable')

    monkeypatch.setattr(evaluator, 'select_relevant_chunks_dense_with_metadata', unavailable)

    with pytest.raises(evaluator.GenerationConfigError) as exc_info:
        evaluator.retrieve_evidence(
            case,
            DENSE_RETRIEVAL_METHOD,
            dense_mode=evaluator.DENSE_MODE_REAL,
        )

    assert 'forbids fallback' in str(exc_info.value)


def test_retrieval_only_never_invokes_generation(monkeypatch):
    def fail_generation_factory(*args, **kwargs):
        raise AssertionError('retrieval-only should not configure generation')

    monkeypatch.setattr(evaluator, 'real_generation_function_factory', fail_generation_factory)

    report = evaluator.evaluate_retrieval_only(
        method_names=[LIGHTRAG_RETRIEVAL_METHOD],
        case_id='single_doc_exact_climate_en',
        dense_mode=evaluator.DENSE_MODE_STUB,
    )

    assert report['real_llm_called'] is False
    assert report['result_count'] == 1
    assert report['results'][0]['retrieval_method'] == LIGHTRAG_RETRIEVAL_METHOD


def test_retrieval_only_stub_dense_selected_case_metadata():
    report = evaluator.evaluate_retrieval_only(
        method_names=[DENSE_RETRIEVAL_METHOD],
        case_id='broad_synthesis_humanitarian_speech',
        dense_mode=evaluator.DENSE_MODE_STUB,
    )

    result = report['results'][0]
    assert result['case_id'] == 'broad_synthesis_humanitarian_speech'
    assert result['retrieval_method'] == DENSE_RETRIEVAL_METHOD
    assert result['dense_mode'] == evaluator.DENSE_MODE_STUB
    assert result['ranked_source_documents']
    assert result['chunk_count'] > 0
    assert result['evidence_character_count'] > 0


def test_generation_result_metadata_includes_dense_mode():
    case = evaluator.load_generation_cases()[0]

    result = evaluator.evaluate_case_method(
        case,
        DENSE_RETRIEVAL_METHOD,
        dense_mode=evaluator.DENSE_MODE_STUB,
    )

    assert result['dense_mode'] == evaluator.DENSE_MODE_STUB
