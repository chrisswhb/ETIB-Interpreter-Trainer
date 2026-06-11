# RAG Retrieval Evaluation Summary

## Purpose

This benchmark compares retrieval strategies for document-grounded interpreter training in Arabic, French, and English. The goal is to identify which retrieval methods are strongest for exact terminology, numbers, names, multilingual matching, paraphrase-like queries, and relation-style questions before any retrieval method is integrated into production Flask endpoints.

## Methods Tested

- `rag_lite_keyword_metadata`
- `bm25_keyword`
- `dense_multilingual_embedding`
- `hybrid_bm25_dense`
- Hybrid weight variants:
  - `hybrid_bm25_dense_0_2_0_8`: 0.2 BM25 / 0.8 dense
  - `hybrid_bm25_dense_0_3_0_7`: 0.3 BM25 / 0.7 dense
  - `hybrid_bm25_dense_0_4_0_6`: 0.4 BM25 / 0.6 dense
  - `hybrid_bm25_dense`: 0.5 BM25 / 0.5 dense

## Benchmark Categories

- `english_exact`
- `french_accent`
- `arabic_normalization`
- `numbers_names`
- `noisy_ranking`
- `rare_terminology`
- `semantic_paraphrase`
- `cross_language`
- `graph_relation_placeholder`

## Final Metrics

| Method | Recall@1 | Recall@3 | MRR | Exact Hit | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rag_lite_keyword_metadata` | 0.64 | 0.79 | 0.70 | 0.91 | 0.63 ms |
| `bm25_keyword` | 0.64 | 0.79 | 0.70 | 0.91 | 0.74 ms |
| `dense_multilingual_embedding` | 0.79 | 0.86 | 0.82 | 0.98 | 893.31 ms |
| `hybrid_bm25_dense_0_2_0_8` | 0.64 | 0.86 | 0.75 | 0.98 | 143.10 ms |
| `hybrid_bm25_dense_0_3_0_7` | 0.64 | 0.86 | 0.75 | 0.98 | 167.75 ms |
| `hybrid_bm25_dense_0_4_0_6` | 0.64 | 0.86 | 0.75 | 0.98 | 169.05 ms |
| `hybrid_bm25_dense` | 0.64 | 0.86 | 0.75 | 0.98 | 165.22 ms |

## Main Findings

- Dense multilingual retrieval achieved the best overall quality.
- Dense improved Recall@1, Recall@3, MRR, and exact hit rate compared with lexical methods.
- Dense improved cross-language and rare-terminology retrieval.
- BM25 did not outperform the RAG-lite baseline on this benchmark.
- Hybrid methods matched dense on Recall@3 and exact hit rate, but did not beat dense on Recall@1 or MRR.
- Hybrid methods were faster than dense in these runs but still slower than lexical methods.
- Semantic paraphrase remained unresolved for all tested methods.

## Recommended Current Winner

For retrieval quality, `dense_multilingual_embedding` is the current best method.

For speed and simplicity, `rag_lite_keyword_metadata` or `bm25_keyword` remain best.

For balanced future work, hybrid retrieval is promising but needs better ranking and tuning.

## What Not To Claim

- Do not claim BM25 improved retrieval quality.
- Do not claim hybrid beats dense.
- Do not claim dense solved semantic paraphrase.
- Do not claim GraphRAG or LightRAG was implemented.

## Next Recommended Work

- Improve the semantic paraphrase benchmark or test a stronger multilingual embedding model.
- Consider better hybrid ranking, such as dense-first retrieval with BM25 tie-breaking.
- Later integrate the winning retrieval method into a Flask endpoint only after team agreement.
- Keep GraphRAG and LightRAG as future comparison work, not current implementation.
