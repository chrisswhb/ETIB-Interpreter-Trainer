"""
Offline GraphRAG-style retrieval for Phase 2 relation evaluation.

This is a deterministic in-memory prototype, not the official Microsoft
GraphRAG package. It combines a global theme/community layer with local
relation-path scoring over chunk, document, concept, entity, and relation nodes.
It does not call an LLM, use a graph database, or touch production endpoints.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

from utils.document_grounding import _tokenize
from utils.lightrag_retrieval import (
    extract_entities,
    extract_query_concepts,
    extract_relation_concepts,
    extract_relation_cues,
)


GRAPHRAG_RETRIEVAL_METHOD = 'graphrag_global_local'

THEME_DEFINITIONS = {
    'climate_agriculture': {
        'climate disruption',
        'agricultural production',
    },
    'food_migration': {
        'food insecurity',
        'migration pressure',
    },
    'health_displacement': {
        'migration pressure',
        'health services',
        'vaccination programmes',
    },
    'regional_funding': {
        'regional funding',
        'World Health Organization',
        'Arab League',
        'United Nations Development Programme',
    },
    'regional_cooperation': {
        'regional cooperation',
        'logistics',
        'vulnerable communities',
    },
}

QUERY_THEME_ALIASES = {
    'climate_agriculture': {'climate', 'agriculture', 'agricultural'},
    'food_migration': {'food', 'food security', 'migration', 'displacement'},
    'health_displacement': {'health', 'vaccination', 'displacement'},
    'regional_funding': {'funding', 'fund', 'resilience', 'organizations', 'partners'},
    'regional_cooperation': {'regional cooperation', 'cooperation', 'logistics'},
}

RELATION_EDGE_TYPES = {
    'cause': 'causal_link',
    'support': 'funding_support_link',
    'funding': 'funding_support_link',
    'coordination': 'coordination_link',
}


def build_global_local_graph(chunk_records: list[dict]) -> dict:
    """Build a typed graph with document, chunk, concept, entity, relation, and theme nodes."""
    nodes = {}
    edges = []
    chunks = []
    concept_to_chunks = defaultdict(set)
    document_to_chunks = defaultdict(set)
    theme_to_documents = defaultdict(set)
    theme_to_chunks = defaultdict(set)
    adjacency = defaultdict(list)

    for chunk_position, record in enumerate(chunk_records):
        document = record.get('source_filename')
        text = record.get('text', '')
        concepts = extract_relation_concepts(text)
        entities = extract_entities(text)
        relation_cues = extract_relation_cues(text)
        themes = assign_themes(concepts | entities)

        document_node = f'document:{document}'
        chunk_node = f'chunk:{chunk_position}'
        nodes.setdefault(document_node, {'type': 'document', 'name': document})
        nodes[chunk_node] = {
            'type': 'chunk',
            'source_filename': document,
            'chunk_index': record.get('chunk_index', 0),
        }
        _add_edge(edges, adjacency, document_node, chunk_node, 'contains_chunk', 1.0)
        document_to_chunks[document].add(chunk_position)

        chunk = {
            'position': chunk_position,
            'record': record,
            'concepts': concepts,
            'entities': entities,
            'relation_cues': relation_cues,
            'themes': themes,
            'chunk_node': chunk_node,
            'document_node': document_node,
        }
        chunks.append(chunk)

        for concept in concepts:
            concept_node = f'concept:{concept}'
            nodes.setdefault(concept_node, {'type': 'concept', 'name': concept})
            concept_to_chunks[concept].add(chunk_position)
            _add_edge(edges, adjacency, chunk_node, concept_node, 'mentions_concept', 1.0)

        for entity in entities:
            entity_node = f'entity:{entity}'
            nodes.setdefault(entity_node, {'type': 'entity', 'name': entity})
            concept_to_chunks[entity].add(chunk_position)
            _add_edge(edges, adjacency, chunk_node, entity_node, 'mentions_entity', 1.0)

        for cue in relation_cues:
            relation_node = f'relation:{cue}'
            nodes.setdefault(relation_node, {'type': 'relation', 'name': cue})
            _add_edge(edges, adjacency, chunk_node, relation_node, 'has_relation_cue', 1.0)
            _add_edge(
                edges,
                adjacency,
                chunk_node,
                relation_node,
                RELATION_EDGE_TYPES.get(cue, 'relation_link'),
                1.5,
            )

        for theme in themes:
            theme_node = f'theme:{theme}'
            nodes.setdefault(theme_node, {'type': 'theme', 'name': theme})
            theme_to_documents[theme].add(document)
            theme_to_chunks[theme].add(chunk_position)
            _add_edge(edges, adjacency, theme_node, document_node, 'theme_in_document', 1.0)
            _add_edge(edges, adjacency, theme_node, chunk_node, 'theme_supports_chunk', 1.0)

        for left, right in itertools.combinations(sorted(concepts | entities), 2):
            left_node = _concept_or_entity_node(left, concepts)
            right_node = _concept_or_entity_node(right, concepts)
            _add_edge(edges, adjacency, left_node, right_node, 'co_occurs', 0.5)

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': dict(adjacency),
        'chunks': chunks,
        'concept_to_chunks': dict(concept_to_chunks),
        'document_to_chunks': dict(document_to_chunks),
        'theme_to_documents': dict(theme_to_documents),
        'theme_to_chunks': dict(theme_to_chunks),
    }


def assign_themes(concepts: set[str]) -> set[str]:
    """Assign high-level deterministic communities from concepts/entities."""
    themes = set()
    if 'climate disruption' in concepts:
        themes.add('climate_agriculture')
    if concepts.intersection({'food insecurity', 'migration pressure'}):
        themes.add('food_migration')
    if concepts.intersection({'health services', 'vaccination programmes'}):
        themes.add('health_displacement')
    if concepts.intersection(THEME_DEFINITIONS['regional_funding']):
        themes.add('regional_funding')
    if concepts.intersection(THEME_DEFINITIONS['regional_cooperation']):
        themes.add('regional_cooperation')
    return themes


def extract_query_themes(query: str, query_concepts: set[str]) -> set[str]:
    normalized_query = ' '.join((query or '').lower().replace('-', ' ').split())
    themes = assign_themes(query_concepts)
    for theme, aliases in QUERY_THEME_ALIASES.items():
        if any(alias in normalized_query for alias in aliases):
            themes.add(theme)
    return themes


def select_relevant_chunks_graphrag_with_metadata(
    chunk_records: list[dict],
    params: dict,
    max_excerpts: int,
    max_total_characters: int,
) -> list[dict]:
    """Select evidence with global theme coverage plus local graph paths."""
    if not chunk_records:
        return []

    graph = build_global_local_graph(chunk_records)
    query = ' '.join(
        str(params.get(key) or '')
        for key in ('query', 'domain', 'scenario', 'language')
    )
    query_terms = _tokenize(query)
    query_concepts = extract_query_concepts(query)
    query_relations = extract_relation_cues(str(params.get('query') or query))
    query_themes = extract_query_themes(query, query_concepts)
    global_documents = rank_global_documents(graph, query_themes, query_concepts)

    scored_chunks = [
        score_global_local_chunk(
            chunk,
            query_terms=query_terms,
            query_concepts=query_concepts,
            query_relations=query_relations,
            query_themes=query_themes,
            global_documents=global_documents,
        )
        for chunk in graph['chunks']
    ]
    selected = select_global_local_evidence(
        scored_chunks,
        query_themes=query_themes,
        max_excerpts=max_excerpts,
        max_total_characters=max_total_characters,
    )
    return [format_graphrag_chunk(chunk) for chunk in selected]


def rank_global_documents(
    graph: dict,
    query_themes: set[str],
    query_concepts: set[str],
) -> dict[str, float]:
    """Rank documents through global themes and concept coverage."""
    scores = Counter()
    for theme in query_themes:
        for document in graph.get('theme_to_documents', {}).get(theme, set()):
            scores[document] += 4.0

    for concept in query_concepts:
        for chunk_index in graph.get('concept_to_chunks', {}).get(concept, set()):
            chunk = graph['chunks'][chunk_index]
            scores[chunk['record'].get('source_filename')] += 1.5

    return dict(scores)


def score_global_local_chunk(
    chunk: dict,
    query_terms: set[str],
    query_concepts: set[str],
    query_relations: set[str],
    query_themes: set[str],
    global_documents: dict[str, float],
) -> dict:
    record = chunk['record']
    concepts = chunk['concepts'] | chunk['entities']
    themes = chunk['themes']
    relation_cues = chunk['relation_cues']
    direct_concepts = concepts.intersection(query_concepts)
    theme_hits = themes.intersection(query_themes)
    relation_hits = relation_cues.intersection(query_relations)
    lexical_hits = _tokenize(record.get('text', '')).intersection(query_terms)
    document = record.get('source_filename')

    local_path_score = (
        len(direct_concepts) * 6.0
        + len(relation_hits) * 4.0
        + min(6, len(lexical_hits)) * 0.3
    )
    global_score = len(theme_hits) * 5.0 + global_documents.get(document, 0)
    synthesis_bonus = 0.0
    if len(query_themes) >= 4 and theme_hits:
        synthesis_bonus = 3.0
    if 'cause' in query_relations and 'cause' in relation_cues:
        local_path_score += 18.0
    if len(query_themes) >= 4 and 'cause' in relation_cues:
        local_path_score += 12.0

    scored = dict(chunk)
    scored.update({
        'score': global_score + local_path_score + synthesis_bonus,
        'global_score': global_score,
        'local_path_score': local_path_score,
        'matched_themes': theme_hits,
        'direct_concepts': direct_concepts,
        'relation_hits': relation_hits,
        'expanded_concepts': expand_local_paths(concepts, query_concepts),
    })
    return scored


def expand_local_paths(chunk_concepts: set[str], query_concepts: set[str]) -> set[str]:
    """Return concepts that bridge local chunk evidence with query seeds."""
    expanded = set(chunk_concepts.intersection(query_concepts))
    for theme, concepts in THEME_DEFINITIONS.items():
        if chunk_concepts.intersection(concepts) and query_concepts.intersection(concepts):
            expanded.update(concepts.intersection(chunk_concepts))
    return expanded


def select_global_local_evidence(
    scored_chunks: list[dict],
    query_themes: set[str],
    max_excerpts: int,
    max_total_characters: int,
) -> list[dict]:
    """Select source-diverse evidence balancing global themes and local paths."""
    selected = []
    selected_sources = set()
    covered_themes = set()
    total_chars = 0
    candidates = sorted(
        scored_chunks,
        key=lambda item: (
            -item['score'],
            item['record'].get('source_order', 0),
            item['record'].get('chunk_index', 0),
        ),
    )

    while candidates and len(selected) < max_excerpts:
        best_index = 0
        best_value = None
        for index, candidate in enumerate(candidates):
            record = candidate['record']
            text_length = len(record.get('text', ''))
            if total_chars and total_chars + text_length > max_total_characters:
                continue

            source = record.get('source_filename')
            new_themes = candidate['matched_themes'] - covered_themes
            source_bonus = 4.0 if source not in selected_sources else 0
            theme_bonus = min(3, len(new_themes)) * 3.0
            local_bonus = min(3, len(candidate['direct_concepts'])) * 1.0
            value = candidate['score'] + source_bonus + theme_bonus + local_bonus
            sort_tuple = (
                value,
                -record.get('source_order', 0),
                -record.get('chunk_index', 0),
            )
            if best_value is None or sort_tuple > best_value:
                best_value = sort_tuple
                best_index = index

        chosen = candidates.pop(best_index)
        selected.append(chosen)
        selected_sources.add(chosen['record'].get('source_filename'))
        covered_themes.update(chosen['matched_themes'])
        total_chars += len(chosen['record'].get('text', ''))

        if query_themes and covered_themes.issuperset(query_themes) and len(selected) >= max_excerpts:
            break

    return selected


def format_graphrag_chunk(scored_chunk: dict) -> dict:
    record = scored_chunk['record']
    return {
        'text': record.get('text', ''),
        'source_filename': record.get('source_filename'),
        'source_type': record.get('source_type'),
        'chunk_index': record.get('chunk_index', 0),
        'score': scored_chunk.get('score', 0),
        'retrieval_method': GRAPHRAG_RETRIEVAL_METHOD,
        'graph_diagnostics': {
            'matched_themes': sorted(scored_chunk.get('matched_themes', [])),
            'direct_concepts': sorted(scored_chunk.get('direct_concepts', [])),
            'expanded_concepts': sorted(scored_chunk.get('expanded_concepts', [])),
            'relation_evidence': sorted(scored_chunk.get('relation_hits', [])),
            'global_score': scored_chunk.get('global_score', 0),
            'local_path_score': scored_chunk.get('local_path_score', 0),
        },
    }


def _add_edge(edges: list[dict], adjacency: dict, source: str, target: str, edge_type: str, weight: float) -> None:
    edge = {
        'source': source,
        'target': target,
        'type': edge_type,
        'weight': weight,
    }
    edges.append(edge)
    adjacency[source].append(edge)


def _concept_or_entity_node(label: str, concepts: set[str]) -> str:
    node_type = 'concept' if label in concepts else 'entity'
    return f'{node_type}:{label}'
