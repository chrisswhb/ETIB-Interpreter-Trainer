"""
Offline LightRAG-style relation retrieval for Phase 2 evaluation.

This is a small deterministic prototype, not the official LightRAG package.
It builds an in-memory concept/entity graph over the provided chunks and uses
local relation cues plus graph expansion to select multi-document evidence.
No LLM, network call, vector database, or production endpoint is involved.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

from utils.document_grounding import _tokenize


LIGHTRAG_RETRIEVAL_METHOD = 'lightrag_relation_graph'

CONCEPT_ALIASES = {
    'climate disruption': {
        'climate disruption',
        'climate stress',
        'heat waves',
        'irregular rainfall',
        'drought',
    },
    'agricultural production': {
        'agricultural production',
        'harvests',
        'cereal yields',
        'small farmers',
        'rural livelihoods',
        'water-saving irrigation',
        'drought-resistant seeds',
    },
    'food insecurity': {
        'food security',
        'food insecurity',
        'food-security',
        'food prices',
        'stable meals',
        'food reserves',
        'grain reserves',
    },
    'migration pressure': {
        'migration',
        'migration pressure',
        'cross-border migration',
        'displacement',
        'displaced populations',
        'temporary displacement',
    },
    'health services': {
        'health services',
        'health workers',
        'host communities',
        'maternal clinics',
        'emergency care',
        'medicine refills',
    },
    'vaccination programmes': {
        'vaccination',
        'vaccination programmes',
        'routine immunisation',
        'immunisation',
        'mobile vaccination teams',
        'cold-chain equipment',
        'vaccination outreach',
    },
    'regional funding': {
        'funding',
        'funds',
        'regional partners',
        'regional resilience fund',
        '42 million dollars',
        'donor',
    },
    'regional cooperation': {
        'regional cooperation',
        'cooperation',
        'coordinate',
        'coordination',
        'joint policy planning',
        'countries',
        'institutions',
        'cross-border support',
    },
    'logistics': {
        'logistics',
        'food-security logistics',
        'customs authorities',
        'agricultural inputs',
    },
    'vulnerable communities': {
        'vulnerable communities',
        'vulnerable populations',
        'families',
        'community leaders',
    },
    'World Health Organization': {'world health organization'},
    'Arab League': {'arab league'},
    'United Nations Development Programme': {
        'united nations development programme',
    },
}

QUERY_CONCEPT_ALIASES = {
    'climate disruption': {'climate', 'climate disruption'},
    'agricultural production': {'agriculture', 'agricultural production'},
    'food insecurity': {'food security', 'food insecurity', 'food-security'},
    'migration pressure': {'migration', 'migration pressure', 'displacement'},
    'health services': {'health', 'health services'},
    'vaccination programmes': {'vaccination', 'vaccination programmes'},
    'regional funding': {'funding', 'fund', 'resilience'},
    'regional cooperation': {'cooperation', 'regional cooperation'},
    'logistics': {'logistics'},
    'vulnerable communities': {'vulnerable communities', 'vulnerable populations'},
}

RELATION_CUES = {
    'cause': {
        'contributes',
        'create',
        'creates',
        'pressure',
        'reducing',
        'reduces',
        'lower',
        'fall',
        'after',
        'when',
        'chain',
        'connected',
        'link',
    },
    'support': {
        'support',
        'supports',
        'protect',
        'needed',
        'reduce',
        'outreach',
        'programmes',
        'assistance',
    },
    'funding': {
        'fund',
        'funds',
        'funding',
        'allocated',
        'partners',
        'million',
        'dollars',
    },
    'coordination': {
        'coordinate',
        'coordination',
        'cooperation',
        'logistics',
        'cross-border',
        'policy',
        'joint',
    },
    'health': {
        'health',
        'vaccination',
        'immunisation',
        'clinics',
        'medicine',
    },
}

STOP_ENTITIES = {
    'When',
    'The',
    'In',
    'No',
    'This',
    'Phase',
}


def build_relation_graph(chunk_records: list[dict]) -> dict:
    """Build an in-memory graph of concepts, entities, chunks, and documents."""
    chunks = []
    concept_to_chunks = defaultdict(set)
    concept_to_documents = defaultdict(set)
    edges = Counter()

    for index, record in enumerate(chunk_records):
        text = record.get('text', '')
        concepts = extract_relation_concepts(text)
        entities = extract_entities(text)
        relation_cues = extract_relation_cues(text)
        document = record.get('source_filename')
        chunk_node = f'chunk:{index}'
        document_node = f'document:{document}'

        chunk = {
            'index': index,
            'record': record,
            'concepts': concepts,
            'entities': entities,
            'relation_cues': relation_cues,
            'chunk_node': chunk_node,
            'document_node': document_node,
        }
        chunks.append(chunk)

        for concept in concepts | entities:
            concept_to_chunks[concept].add(index)
            if document:
                concept_to_documents[concept].add(document)

        for left, right in itertools.combinations(sorted(concepts | entities), 2):
            edges[(left, right)] += 1
            edges[(right, left)] += 1

    adjacency = defaultdict(dict)
    for (left, right), weight in edges.items():
        adjacency[left][right] = weight

    return {
        'chunks': chunks,
        'concept_to_chunks': dict(concept_to_chunks),
        'concept_to_documents': dict(concept_to_documents),
        'adjacency': dict(adjacency),
    }


def extract_relation_concepts(text: str) -> set[str]:
    """Extract canonical policy concepts from text with deterministic phrases."""
    normalized = _normalize_phrase_text(text)
    concepts = set()
    for concept, aliases in CONCEPT_ALIASES.items():
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            concepts.add(concept)
    return concepts


def extract_query_concepts(text: str) -> set[str]:
    """Extract broader query concepts without over-labeling document chunks."""
    normalized = _normalize_phrase_text(text)
    concepts = extract_relation_concepts(text)
    for concept, aliases in QUERY_CONCEPT_ALIASES.items():
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            concepts.add(concept)
    return concepts


def extract_relation_cues(text: str) -> set[str]:
    tokens = _tokenize(text)
    cues = set()
    for cue_name, cue_tokens in RELATION_CUES.items():
        normalized_cues = set()
        for cue_token in cue_tokens:
            normalized_cues.update(_tokenize(cue_token))
        if tokens.intersection(normalized_cues):
            cues.add(cue_name)
    return cues


def extract_entities(text: str) -> set[str]:
    """Extract simple title-case organization/person-style spans."""
    entities = set()
    for match in re.finditer(r'\b(?:[A-Z][a-z]+(?:\s+|$)){2,5}', text or ''):
        entity = ' '.join(match.group(0).split())
        if entity and entity.split()[0] not in STOP_ENTITIES:
            entities.add(entity)
    return entities


def select_relevant_chunks_lightrag_with_metadata(
    chunk_records: list[dict],
    params: dict,
    max_excerpts: int,
    max_total_characters: int,
) -> list[dict]:
    """Select relation-aware chunks using graph expansion and source diversity."""
    if not chunk_records:
        return []

    graph = build_relation_graph(chunk_records)
    query = ' '.join(
        str(params.get(key) or '')
        for key in ('query', 'domain', 'scenario', 'language')
    )
    query_text = str(params.get('query') or '')
    query_terms = _tokenize(query)
    query_concepts = extract_query_concepts(query)
    query_cues = extract_relation_cues(query_text)
    expanded_concepts = expand_query_concepts(graph, query_concepts)

    scored = [
        _score_chunk(chunk, query_terms, query_concepts, expanded_concepts, query_cues)
        for chunk in graph['chunks']
    ]
    scored.sort(key=lambda item: (
        -item['score'],
        item['record'].get('source_order', 0),
        item['record'].get('chunk_index', 0),
    ))

    selected = _select_diverse_relation_chunks(
        scored,
        max_excerpts=max_excerpts,
        max_total_characters=max_total_characters,
    )
    return [_format_selected_chunk(item) for item in selected]


def expand_query_concepts(graph: dict, query_concepts: set[str]) -> set[str]:
    """Expand query seeds through one graph hop for LightRAG-style retrieval."""
    expanded = set(query_concepts)
    adjacency = graph.get('adjacency', {})
    for concept in query_concepts:
        neighbors = adjacency.get(concept, {})
        top_neighbors = sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))[:4]
        expanded.update(neighbor for neighbor, _ in top_neighbors)
    return expanded


def _score_chunk(
    chunk: dict,
    query_terms: set[str],
    query_concepts: set[str],
    expanded_concepts: set[str],
    query_cues: set[str],
) -> dict:
    text = chunk['record'].get('text', '')
    text_terms = _tokenize(text)
    concepts = chunk['concepts']
    relation_cues = chunk['relation_cues']

    direct_concept_hits = concepts.intersection(query_concepts)
    expanded_concept_hits = concepts.intersection(expanded_concepts)
    cue_hits = relation_cues.intersection(query_cues)
    lexical_hits = text_terms.intersection(query_terms)
    causal_chain_bonus = 0.0
    if 'cause' in query_cues and 'cause' in relation_cues:
        causal_chain_bonus = 30.0

    score = (
        len(direct_concept_hits) * 8.0
        + len(expanded_concept_hits - direct_concept_hits) * 3.5
        + len(cue_hits) * 2.0
        + causal_chain_bonus
        + min(6, len(lexical_hits)) * 0.4
        + min(3, len(concepts)) * 0.25
    )

    scored = dict(chunk)
    scored['score'] = score
    scored['direct_concept_hits'] = direct_concept_hits
    scored['expanded_concept_hits'] = expanded_concept_hits
    scored['cue_hits'] = cue_hits
    return scored


def _select_diverse_relation_chunks(
    scored_chunks: list[dict],
    max_excerpts: int,
    max_total_characters: int,
) -> list[dict]:
    selected = []
    selected_sources = set()
    covered_concepts = set()
    total_chars = 0

    candidates = list(scored_chunks)
    while candidates and len(selected) < max_excerpts:
        best_index = 0
        best_value = None
        for index, candidate in enumerate(candidates):
            record = candidate['record']
            text_length = len(record.get('text', ''))
            if total_chars and total_chars + text_length > max_total_characters:
                continue

            source = record.get('source_filename')
            new_concepts = candidate['concepts'] - covered_concepts
            diversity_bonus = 2.5 if source not in selected_sources else 0
            concept_bonus = min(4, len(new_concepts)) * 1.25
            value = candidate['score'] + diversity_bonus + concept_bonus
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
        covered_concepts.update(chosen['concepts'])
        total_chars += len(chosen['record'].get('text', ''))

    return selected


def _format_selected_chunk(scored_chunk: dict) -> dict:
    record = scored_chunk['record']
    return {
        'text': record.get('text', ''),
        'source_filename': record.get('source_filename'),
        'source_type': record.get('source_type'),
        'chunk_index': record.get('chunk_index', 0),
        'score': scored_chunk.get('score', 0),
        'retrieval_method': LIGHTRAG_RETRIEVAL_METHOD,
        'graph_signals': {
            'concepts': sorted(scored_chunk.get('concepts', [])),
            'entities': sorted(scored_chunk.get('entities', [])),
            'relation_cues': sorted(scored_chunk.get('relation_cues', [])),
            'direct_concept_hits': sorted(scored_chunk.get('direct_concept_hits', [])),
            'expanded_concept_hits': sorted(scored_chunk.get('expanded_concept_hits', [])),
        },
    }


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_phrase_text(phrase)
    return normalized_phrase in normalized_text


def _normalize_phrase_text(text: str) -> str:
    return ' '.join((text or '').lower().replace('-', ' ').split())
