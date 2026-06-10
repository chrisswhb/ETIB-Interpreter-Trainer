"""
Optional dense embedding retrieval helpers for offline evaluation only.

This module intentionally does not integrate with Flask routes or any vector
database. sentence-transformers is imported lazily so lexical evaluators keep
working when dense dependencies or model files are unavailable.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from math import sqrt


DENSE_RETRIEVAL_METHOD = 'dense_multilingual_embedding'
HYBRID_RETRIEVAL_METHOD = 'hybrid_bm25_dense'
DEFAULT_DENSE_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_HYBRID_BM25_WEIGHT = 0.5
DEFAULT_HYBRID_DENSE_WEIGHT = 0.5
OPTIONAL_DEPENDENCY_MESSAGE = (
    'Install optional RAG dependencies with: '
    'python -m pip install -r backend/requirements-rag-optional.txt'
)


class DenseEmbeddingUnavailable(RuntimeError):
    """Raised when optional dense embedding evaluation cannot run."""


def is_dense_embedding_available() -> bool:
    """Return True when sentence-transformers can be imported."""
    return find_spec('sentence_transformers') is not None


def select_relevant_chunks_dense_with_metadata(
    chunk_records: list[dict],
    params: dict,
    max_excerpts: int,
    max_total_characters: int,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
) -> list[dict]:
    """Select chunks by cosine similarity between query and chunk embeddings."""
    if not chunk_records:
        return []

    query = str(params.get('query') or '').strip()
    if not query:
        query = ' '.join(
            str(params.get(key) or '')
            for key in ('domain', 'scenario', 'difficulty', 'mode', 'language')
        ).strip()

    if not query:
        return []

    scored_chunks = score_chunks_dense(chunk_records, params, model_name=model_name)
    scored_chunks.sort(key=lambda item: (
        -item.get('score', 0),
        item.get('source_order', 0),
        item.get('chunk_index', 0),
    ))

    return _select_scored_chunks(
        scored_chunks,
        max_excerpts=max_excerpts,
        max_total_characters=max_total_characters,
        retrieval_method=DENSE_RETRIEVAL_METHOD,
    )


def score_chunks_dense(
    chunk_records: list[dict],
    params: dict,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
) -> list[dict]:
    """Score all chunk records by dense query/chunk cosine similarity."""
    if not chunk_records:
        return []

    query = str(params.get('query') or '').strip()
    if not query:
        query = ' '.join(
            str(params.get(key) or '')
            for key in ('domain', 'scenario', 'difficulty', 'mode', 'language')
        ).strip()

    if not query:
        return []

    model = _load_sentence_transformer(model_name)
    chunk_texts = [record.get('text', '') for record in chunk_records]
    embeddings = model.encode(
        [query] + chunk_texts,
        convert_to_numpy=False,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    query_embedding = _as_float_list(embeddings[0])
    scored_chunks = []
    for record, embedding in zip(chunk_records, embeddings[1:]):
        scored_record = dict(record)
        scored_record['score'] = _cosine_similarity(
            query_embedding,
            _as_float_list(embedding),
        )
        scored_record['retrieval_method'] = DENSE_RETRIEVAL_METHOD
        scored_chunks.append(scored_record)

    return scored_chunks


def select_relevant_chunks_hybrid_with_metadata(
    chunk_records: list[dict],
    params: dict,
    max_excerpts: int,
    max_total_characters: int,
    bm25_weight: float = DEFAULT_HYBRID_BM25_WEIGHT,
    dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
) -> list[dict]:
    """Select chunks with a weighted normalized BM25 + dense score."""
    scored_chunks = score_chunks_hybrid(
        chunk_records,
        params,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        model_name=model_name,
    )
    scored_chunks.sort(key=lambda item: (
        -item.get('score', 0),
        item.get('source_order', 0),
        item.get('chunk_index', 0),
    ))

    return _select_scored_chunks(
        scored_chunks,
        max_excerpts=max_excerpts,
        max_total_characters=max_total_characters,
        retrieval_method=HYBRID_RETRIEVAL_METHOD,
    )


def score_chunks_hybrid(
    chunk_records: list[dict],
    params: dict,
    bm25_weight: float = DEFAULT_HYBRID_BM25_WEIGHT,
    dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
    model_name: str = DEFAULT_DENSE_MODEL_NAME,
) -> list[dict]:
    """Score chunks by combining normalized BM25 and dense scores."""
    if not chunk_records:
        return []

    from utils.document_grounding import score_chunks_bm25

    bm25_chunks = score_chunks_bm25(chunk_records, params)
    dense_chunks = score_chunks_dense(chunk_records, params, model_name=model_name)
    bm25_scores = [chunk.get('score', 0) for chunk in bm25_chunks]
    dense_scores = [chunk.get('score', 0) for chunk in dense_chunks]
    hybrid_scores = combine_normalized_scores(
        bm25_scores,
        dense_scores,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    scored_chunks = []
    for record, bm25_score, dense_score, hybrid_score in zip(
        chunk_records,
        bm25_scores,
        dense_scores,
        hybrid_scores,
    ):
        scored_record = dict(record)
        scored_record['score'] = hybrid_score
        scored_record['retrieval_method'] = HYBRID_RETRIEVAL_METHOD
        scored_record['component_scores'] = {
            'bm25': bm25_score,
            'dense': dense_score,
        }
        scored_chunks.append(scored_record)

    return scored_chunks


def combine_normalized_scores(
    bm25_scores: list[float],
    dense_scores: list[float],
    bm25_weight: float = DEFAULT_HYBRID_BM25_WEIGHT,
    dense_weight: float = DEFAULT_HYBRID_DENSE_WEIGHT,
) -> list[float]:
    """Combine min-max normalized BM25 and dense score lists."""
    normalized_bm25 = _min_max_normalize(bm25_scores)
    normalized_dense = _min_max_normalize(dense_scores)
    return [
        (bm25_weight * bm25_score) + (dense_weight * dense_score)
        for bm25_score, dense_score in zip(normalized_bm25, normalized_dense)
    ]


def _select_scored_chunks(
    scored_chunks: list[dict],
    max_excerpts: int,
    max_total_characters: int,
    retrieval_method: str,
) -> list[dict]:
    """Return selected chunk dictionaries in evaluator response shape."""

    selected = []
    total_chars = 0
    for chunk in scored_chunks:
        if len(selected) >= max_excerpts:
            break

        text_length = len(chunk.get('text', ''))
        if total_chars and total_chars + text_length > max_total_characters:
            continue

        selected_chunk = {
            'text': chunk.get('text', ''),
            'source_filename': chunk.get('source_filename'),
            'source_type': chunk.get('source_type'),
            'chunk_index': chunk.get('chunk_index', 0),
            'score': chunk.get('score', 0),
            'retrieval_method': retrieval_method,
        }
        if 'component_scores' in chunk:
            selected_chunk['component_scores'] = chunk['component_scores']

        selected.append(selected_chunk)
        total_chars += text_length

    return selected


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise DenseEmbeddingUnavailable(
            'sentence-transformers is not installed or could not be imported. '
            f'{OPTIONAL_DEPENDENCY_MESSAGE}'
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise DenseEmbeddingUnavailable(
            f'Could not load dense embedding model {model_name!r}. '
            'Ensure the model is available locally or downloadable. '
            f'{OPTIONAL_DEPENDENCY_MESSAGE}'
        ) from exc


def _as_float_list(vector) -> list[float]:
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [0.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]
