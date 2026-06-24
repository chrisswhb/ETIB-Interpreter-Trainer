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
    HYBRID_METHOD_NAME,
    HYBRID_WEIGHT_CONFIGS,
    METHOD_NAME,
    PHASE2_DENSE_METHOD,
    PHASE2_GRAPHRAG_METHOD,
    PHASE2_LIGHTRAG_METHOD,
    PHASE2_METHODS,
    load_cases,
    load_relation_chunk_records,
    load_relation_cases,
    run_all_phase2_relation_evaluations,
    run_all_evaluations,
    run_evaluation,
    run_phase2_relation_evaluation,
)
from utils.embedding_retrieval import (  # noqa: E402
    combine_normalized_scores,
)
from utils.lightrag_retrieval import (  # noqa: E402
    LIGHTRAG_RETRIEVAL_METHOD,
    build_relation_graph,
    extract_relation_concepts,
    select_relevant_chunks_lightrag_with_metadata,
)
from utils.graphrag_retrieval import (  # noqa: E402
    GRAPHRAG_RETRIEVAL_METHOD,
    assign_themes,
    build_global_local_graph,
    extract_query_themes,
    rank_global_documents,
    select_relevant_chunks_graphrag_with_metadata,
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

    assert set(by_method) == {
        METHOD_NAME,
        BM25_METHOD_NAME,
        DENSE_METHOD_NAME,
        *HYBRID_WEIGHT_CONFIGS,
    }
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

    hybrid_report = by_method[HYBRID_METHOD_NAME]
    assert 'method_available' in hybrid_report
    assert hybrid_report['metrics']['total_cases'] > 0
    assert 'per_case_results' in hybrid_report
    assert hybrid_report['per_case_results']


def test_hybrid_weight_configuration_names_are_registered():
    import scripts.evaluate_retrieval as evaluator

    assert HYBRID_WEIGHT_CONFIGS == {
        'hybrid_bm25_dense_0_2_0_8': (0.2, 0.8),
        'hybrid_bm25_dense_0_3_0_7': (0.3, 0.7),
        'hybrid_bm25_dense_0_4_0_6': (0.4, 0.6),
        HYBRID_METHOD_NAME: (0.5, 0.5),
    }
    for method_name, (bm25_weight, dense_weight) in HYBRID_WEIGHT_CONFIGS.items():
        assert method_name in evaluator.METHODS
        assert evaluator.METHODS[method_name]['bm25_weight'] == bm25_weight
        assert evaluator.METHODS[method_name]['dense_weight'] == dense_weight


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


def test_hybrid_evaluator_skips_cleanly_when_dense_unavailable(monkeypatch):
    import scripts.evaluate_retrieval as evaluator

    for method_name in HYBRID_WEIGHT_CONFIGS:
        monkeypatch.setitem(
            evaluator.METHODS[method_name],
            'availability_check',
            lambda: False,
        )

        report = run_evaluation(method_name=method_name, write_output=False)
        metrics = report['metrics']

        assert report['method_name'] == method_name
        assert report['method_available'] is False
        assert 'sentence-transformers' in report['unavailable_reason']
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


def test_hybrid_score_combination_normalizes_component_ranges():
    combined = combine_normalized_scores(
        bm25_scores=[0, 5, 10],
        dense_scores=[0.2, 0.4, 0.4],
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    assert combined == [0.0, 0.75, 1.0]


def test_hybrid_evaluator_output_shape_when_selector_is_available(monkeypatch):
    import scripts.evaluate_retrieval as evaluator

    def fake_hybrid_selector(
        records,
        params,
        max_excerpts,
        max_total_characters,
        bm25_weight=0.5,
        dense_weight=0.5,
    ):
        selected = []
        for record in records[:max_excerpts]:
            selected.append({
                'text': record['text'],
                'source_filename': record['source_filename'],
                'source_type': record['source_type'],
                'chunk_index': record['chunk_index'],
                'score': 0.8,
                'retrieval_method': HYBRID_METHOD_NAME,
                'component_scores': {
                    'bm25': 1.0,
                    'dense': 0.6,
                },
            })
        return selected

    monkeypatch.setitem(
        evaluator.METHODS[HYBRID_METHOD_NAME],
        'availability_check',
        lambda: True,
    )
    monkeypatch.setitem(
        evaluator.METHODS[HYBRID_METHOD_NAME],
        'selector',
        fake_hybrid_selector,
    )

    report = run_evaluation(method_name=HYBRID_METHOD_NAME, write_output=False)
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
        'component_scores',
    }
    assert first_selected['retrieval_method'] == HYBRID_METHOD_NAME
    assert set(first_selected['component_scores']) == {'bm25', 'dense'}


def test_phase2_relation_cases_schema_is_separate_from_phase1():
    phase1_cases = load_cases()
    relation_cases = load_relation_cases()

    assert len(relation_cases) == 5
    assert {case['category'] for case in relation_cases} == {'relation_multidoc'}
    assert all(case.get('multi_document_required') is True for case in relation_cases)
    assert all(len(case['required_sources']) >= 2 for case in relation_cases)
    assert any(len(case['required_sources']) >= 3 for case in relation_cases)
    assert all(case['difficulty'] in {'medium', 'hard'} for case in relation_cases)
    assert all('relation_chain' in case for case in relation_cases)
    assert all('expected_relation_terms' in case for case in relation_cases)
    assert all('expected_evidence_terms' in case for case in relation_cases)
    assert not any(case['category'] == 'relation_multidoc' for case in phase1_cases)


def test_phase2_relation_metrics_with_fake_dense_selector(monkeypatch):
    import scripts.evaluate_retrieval as evaluator

    def fake_relation_selector(records, params, max_excerpts, max_total_characters):
        selected = []
        for record in records[:max_excerpts]:
            selected.append({
                'text': record['text'],
                'source_filename': record['source_filename'],
                'source_type': record['source_type'],
                'chunk_index': record['chunk_index'],
                'score': 0.9,
                'retrieval_method': DENSE_METHOD_NAME,
            })
        return selected

    monkeypatch.setitem(PHASE2_DENSE_METHOD, 'availability_check', lambda: True)
    monkeypatch.setitem(PHASE2_DENSE_METHOD, 'selector', fake_relation_selector)

    report = run_phase2_relation_evaluation(write_output=False)
    metrics = report['metrics']
    first_case = report['per_case_results'][0]

    assert report['benchmark_name'] == 'phase2_relation_heavy_multidocument'
    assert report['method_name'] == DENSE_METHOD_NAME
    assert report['method_available'] is True
    assert metrics['total_cases'] == 5
    assert metrics['evaluated_cases'] == 5
    assert 0 <= metrics['multi_document_recall_at_3'] <= 1
    assert 0 <= metrics['required_source_coverage'] <= 1
    assert 0 <= metrics['relation_term_hit_rate'] <= 1
    assert 0 <= metrics['relation_chain_coverage'] <= 1
    assert metrics['average_distinct_source_documents_top_3'] >= 1
    assert first_case['multi_document_required'] is True
    assert set(first_case['selected_chunks'][0]) == {
        'text',
        'source_filename',
        'source_type',
        'chunk_index',
        'score',
        'retrieval_method',
    }


def test_phase2_relation_evaluator_skips_cleanly_when_dense_unavailable(monkeypatch):
    monkeypatch.setitem(PHASE2_DENSE_METHOD, 'availability_check', lambda: False)

    report = run_phase2_relation_evaluation(write_output=False)
    metrics = report['metrics']

    assert report['method_available'] is False
    assert 'sentence-transformers' in report['unavailable_reason']
    assert 'backend/requirements-rag-optional.txt' in report['unavailable_reason']
    assert metrics['total_cases'] == 5
    assert metrics['evaluated_cases'] == 0
    assert all(result['skipped'] for result in report['per_case_results'])


def test_lightrag_concept_extraction_finds_relation_terms():
    text = (
        'Climate disruption reduces agricultural production. Food insecurity '
        'then contributes to displacement and migration pressure.'
    )

    concepts = extract_relation_concepts(text)

    assert 'climate disruption' in concepts
    assert 'agricultural production' in concepts
    assert 'food insecurity' in concepts
    assert 'migration pressure' in concepts


def test_lightrag_relation_graph_tracks_chunks_and_concepts():
    case = load_relation_cases()[0]
    records = load_relation_chunk_records(case)

    graph = build_relation_graph(records)

    assert len(graph['chunks']) == len(records)
    assert 'food insecurity' in graph['concept_to_chunks']
    assert graph['concept_to_chunks']['food insecurity']
    assert 'climate disruption' in graph['adjacency']


def test_lightrag_retrieval_output_shape():
    case = load_relation_cases()[0]
    records = load_relation_chunk_records(case)
    params = dict(case['params'])
    params['query'] = case['query']

    selected = select_relevant_chunks_lightrag_with_metadata(
        records,
        params,
        max_excerpts=3,
        max_total_characters=3600,
    )
    first_selected = selected[0]

    assert len(selected) == 3
    assert set(first_selected) == {
        'text',
        'source_filename',
        'source_type',
        'chunk_index',
        'score',
        'retrieval_method',
        'graph_signals',
    }
    assert first_selected['retrieval_method'] == LIGHTRAG_RETRIEVAL_METHOD
    assert first_selected['graph_signals']['concepts']


def test_phase2_lightrag_evaluator_runs_without_optional_dense_dependency():
    report = run_phase2_relation_evaluation(
        method_name=LIGHTRAG_RETRIEVAL_METHOD,
        write_output=False,
    )
    metrics = report['metrics']

    assert report['method_name'] == LIGHTRAG_RETRIEVAL_METHOD
    assert report['method_available'] is True
    assert metrics['total_cases'] == 5
    assert metrics['evaluated_cases'] == 5
    assert metrics['multi_document_recall_at_3'] >= 0
    assert report['per_case_results']


def test_phase2_evaluator_registers_dense_lightrag_and_graphrag_methods(monkeypatch):
    monkeypatch.setitem(PHASE2_DENSE_METHOD, 'availability_check', lambda: False)

    reports = run_all_phase2_relation_evaluations(write_output=False)
    by_method = {report['method_name']: report for report in reports}

    assert set(PHASE2_METHODS) == {
        DENSE_METHOD_NAME,
        LIGHTRAG_RETRIEVAL_METHOD,
        GRAPHRAG_RETRIEVAL_METHOD,
    }
    assert set(by_method) == {
        DENSE_METHOD_NAME,
        LIGHTRAG_RETRIEVAL_METHOD,
        GRAPHRAG_RETRIEVAL_METHOD,
    }
    assert by_method[DENSE_METHOD_NAME]['method_available'] is False
    assert by_method[LIGHTRAG_RETRIEVAL_METHOD]['method_available'] is True
    assert by_method[GRAPHRAG_RETRIEVAL_METHOD]['method_available'] is True


def test_graphrag_global_local_graph_has_typed_nodes_and_edges():
    case = load_relation_cases()[0]
    records = load_relation_chunk_records(case)

    graph = build_global_local_graph(records)
    node_types = {node['type'] for node in graph['nodes'].values()}
    edge_types = {edge['type'] for edge in graph['edges']}

    assert {'document', 'chunk', 'concept', 'entity', 'relation', 'theme'}.issubset(node_types)
    assert {
        'contains_chunk',
        'mentions_concept',
        'mentions_entity',
        'has_relation_cue',
        'theme_in_document',
        'theme_supports_chunk',
        'co_occurs',
        'causal_link',
    }.issubset(edge_types)
    assert graph['theme_to_documents']
    assert graph['concept_to_chunks']['food insecurity']


def test_graphrag_global_themes_are_deterministic():
    concepts = {
        'climate disruption',
        'food insecurity',
        'health services',
        'regional funding',
        'regional cooperation',
    }

    assert assign_themes(concepts) == {
        'climate_agriculture',
        'food_migration',
        'health_displacement',
        'regional_funding',
        'regional_cooperation',
    }


def test_graphrag_local_query_theme_and_document_ranking():
    case = next(
        item for item in load_relation_cases()
        if item['id'] == 'full_humanitarian_speech_evidence'
    )
    records = load_relation_chunk_records(case)
    graph = build_global_local_graph(records)
    query_concepts = {
        'climate disruption',
        'food insecurity',
        'migration pressure',
        'health services',
        'regional funding',
    }
    query_themes = extract_query_themes(case['query'], query_concepts)
    ranked_documents = rank_global_documents(graph, query_themes, query_concepts)

    assert {'food_migration', 'health_displacement', 'regional_funding'}.issubset(
        query_themes
    )
    assert ranked_documents['food_security_migration.txt'] > 0
    assert ranked_documents['health_displacement.txt'] > 0
    assert ranked_documents['regional_funding.txt'] > 0


def test_graphrag_retrieval_output_shape():
    case = load_relation_cases()[0]
    records = load_relation_chunk_records(case)
    params = dict(case['params'])
    params['query'] = case['query']

    selected = select_relevant_chunks_graphrag_with_metadata(
        records,
        params,
        max_excerpts=3,
        max_total_characters=3600,
    )
    first_selected = selected[0]

    assert len(selected) == 3
    assert set(first_selected) == {
        'text',
        'source_filename',
        'source_type',
        'chunk_index',
        'score',
        'retrieval_method',
        'graph_diagnostics',
    }
    assert first_selected['retrieval_method'] == GRAPHRAG_RETRIEVAL_METHOD
    assert {
        'matched_themes',
        'direct_concepts',
        'expanded_concepts',
        'relation_evidence',
        'global_score',
        'local_path_score',
    }.issubset(first_selected['graph_diagnostics'])


def test_graphrag_broad_synthesis_case_uses_global_and_local_evidence():
    case = next(
        item for item in load_relation_cases()
        if item['id'] == 'full_humanitarian_speech_evidence'
    )
    records = load_relation_chunk_records(case)
    params = dict(case['params'])
    params['query'] = case['query']

    selected = select_relevant_chunks_graphrag_with_metadata(
        records,
        params,
        max_excerpts=3,
        max_total_characters=3600,
    )
    selected_sources = {chunk['source_filename'] for chunk in selected}
    required_sources = set(case['required_sources'])
    covered_sources = selected_sources.intersection(required_sources)

    assert len(covered_sources) >= 2
    assert 'health_displacement.txt' in selected_sources
    assert any(chunk['graph_diagnostics']['matched_themes'] for chunk in selected)
    assert any(chunk['graph_diagnostics']['direct_concepts'] for chunk in selected)


def test_phase2_graphrag_evaluator_runs():
    report = run_phase2_relation_evaluation(
        method_name=GRAPHRAG_RETRIEVAL_METHOD,
        write_output=False,
    )
    metrics = report['metrics']

    assert report['method_name'] == GRAPHRAG_RETRIEVAL_METHOD
    assert report['method_available'] is True
    assert PHASE2_GRAPHRAG_METHOD['method_name'] == GRAPHRAG_RETRIEVAL_METHOD
    assert metrics['total_cases'] == 5
    assert metrics['evaluated_cases'] == 5
    assert metrics['required_source_coverage'] >= 0.8
    assert report['per_case_results']
