# RAG Retrieval Evaluation Summary

## Purpose

This report tracks retrieval experiments for document-grounded interpreter
training in Arabic, French, and English. Phase 1 measures general multilingual
retrieval quality for terminology, numbers, names, accent normalization,
cross-language matching, and simple relation-style queries. Phase 2 adds a
separate relation-heavy multi-document benchmark to test whether future
LightRAG or GraphRAG experiments are justified for grounded speech generation.

Phase 1 and Phase 2 scores must not be averaged together because they test
different retrieval goals.

## Phase 1: General Multilingual Retrieval

### Methods Tested

- `rag_lite_keyword_metadata`
- `bm25_keyword`
- `dense_multilingual_embedding`
- `hybrid_bm25_dense`
- Hybrid weight variants:
  - `hybrid_bm25_dense_0_2_0_8`: 0.2 BM25 / 0.8 dense
  - `hybrid_bm25_dense_0_3_0_7`: 0.3 BM25 / 0.7 dense
  - `hybrid_bm25_dense_0_4_0_6`: 0.4 BM25 / 0.6 dense
  - `hybrid_bm25_dense`: 0.5 BM25 / 0.5 dense

### Benchmark Categories

- `english_exact`
- `french_accent`
- `arabic_normalization`
- `numbers_names`
- `noisy_ranking`
- `rare_terminology`
- `semantic_paraphrase`
- `cross_language`
- `graph_relation_placeholder`

### Final Phase 1 Metrics

| Method | Recall@1 | Recall@3 | MRR | Exact Hit | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rag_lite_keyword_metadata` | 0.64 | 0.79 | 0.70 | 0.91 | approximately 0.6 ms |
| `bm25_keyword` | 0.64 | 0.79 | 0.70 | 0.91 | approximately 0.7 ms |
| `dense_multilingual_embedding` | 0.79 | 0.86 | 0.82 | 0.98 | approximately 0.9 seconds warm, variable by runtime |
| `hybrid_bm25_dense` variants | 0.64 | 0.86 | 0.75 | 0.98 | approximately 140-170 ms in tested runs |

### Phase 1 Conclusions

- Dense multilingual retrieval achieved the best overall retrieval quality.
- Dense improved Recall@1, Recall@3, MRR, and exact hit rate compared with lexical methods.
- Dense improved cross-language retrieval and rare terminology retrieval.
- BM25 did not outperform the RAG-lite baseline on this benchmark.
- Hybrid methods matched dense on Recall@3 and exact hit rate, but did not beat dense on Recall@1 or MRR.
- Hybrid methods were faster than dense in these runs but still slower than lexical methods.
- Semantic paraphrase remained unresolved for all tested methods.
- Dense is the current production winner for document-grounded speech generation.

## Phase 2: Multi-Document Relation Retrieval

### Purpose

Phase 2 tests whether a speech-generation request can retrieve evidence that
connects causes, organizations, policies, and outcomes across multiple uploaded
documents. These cases are designed to motivate future graph-based comparisons
without implementing LightRAG or GraphRAG yet.

### Corpus

The Phase 2 corpus is fictional, small, and relation-heavy:

- `climate_agriculture.txt`: climate disruption reduces agricultural production and harms rural livelihoods.
- `food_security_migration.txt`: food insecurity contributes to displacement and migration pressure.
- `health_displacement.txt`: displaced populations increase pressure on health services and vaccination programmes.
- `regional_funding.txt`: regional partners and international organizations fund resilience, health, and food-security programmes.
- `cooperation_policy.txt`: countries and institutions coordinate policy, logistics, and cross-border support.

### Phase 2 Cases

There are 5 relation-heavy cases. Each requires evidence from at least two
documents, and one case requires a three-document relation chain.

Example relation chains:

- climate disruption -> agricultural production -> food insecurity -> migration pressure
- food insecurity -> displacement -> health services -> vaccination programmes
- Regional Resilience Fund -> vaccination outreach -> food-security logistics
- regional cooperation -> logistics -> cross-border support -> vulnerable communities
- climate disruption -> agricultural production -> food insecurity -> displacement -> health services

### Phase 2 Metrics

- `multi_document_recall_at_3`: fraction of cases where all required source documents appear in the top 3 chunks.
- `required_source_coverage`: average fraction of required source documents retrieved.
- `relation_term_hit_rate`: average fraction of expected relation terms found in selected chunks.
- `average_distinct_source_documents_top_3`: average number of distinct source documents represented in top 3.
- `relation_chain_coverage`: average fraction of expected relation-chain terms found in selected chunks.

### Dense Baseline Results

The Phase 2 dense baseline is wired into the offline evaluator as
`dense_multilingual_embedding`, using the same optional dense helper as the
production RAG path. The run below used `paraphrase-multilingual-MiniLM-L12-v2`
from the existing local model cache in offline mode. No model download was
needed during the benchmark run.

Latest local Phase 2 dense run:

| Method | Cases | Evaluated | Multi-doc Recall@3 | Required-source coverage | Relation-term hit rate | Relation-chain coverage | Avg distinct sources@3 | Avg latency | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_multilingual_embedding` | 5 | 5 | 0.80 | 0.87 | 0.96 | 0.92 | 3.00 | 1736.45 ms | measured |

Case-level dense coverage:

- `climate_food_migration_chain`: covered. Retrieved `food_security_migration.txt`, `cooperation_policy.txt`, and `climate_agriculture.txt`.
- `displacement_health_service_chain`: covered. Retrieved `food_security_migration.txt`, `health_displacement.txt`, and `regional_funding.txt`.
- `funding_health_food_security`: covered. Retrieved `regional_funding.txt`, `food_security_migration.txt`, and `health_displacement.txt`.
- `regional_cooperation_support_chain`: covered. Retrieved `cooperation_policy.txt`, `regional_funding.txt`, and `health_displacement.txt`.
- `full_humanitarian_speech_evidence`: partially covered. Retrieved `cooperation_policy.txt`, `food_security_migration.txt`, and `regional_funding.txt`, but missed two required source documents for the full three-document chain: `climate_agriculture.txt` and `health_displacement.txt`.

### Current Phase 2 Interpretation

- The benchmark is ready for dense, LightRAG, and GraphRAG comparison.
- Dense covered all medium relation cases and most hard relation terms.
- Dense partially failed the broadest three-document speech-evidence case by retrieving relevant cooperation, migration, and funding evidence while missing the climate-agriculture and health-displacement source requirements.
- The exact weakness Phase 2 exposed is independent chunk ranking that can retrieve relevant individual facts without covering every required document in a long multi-hop relation chain.
- LightRAG or GraphRAG is justified for testing next because graph-style methods may improve required-source coverage and relation-chain completeness on broad multi-document speech-generation requests.

### LightRAG-Style Offline Comparison

`lightrag_relation_graph` is an offline-only local prototype. It does not use
the official LightRAG package, does not call an LLM, does not use a vector
database, and is not integrated into any Flask endpoint. It builds a small
in-memory graph from deterministic concept, entity, relation-cue, chunk, and
document links, then selects diverse relation evidence for the Phase 2 cases.

| Method | Multi-doc Recall@3 | Required-source coverage | Relation-term hit rate | Relation-chain coverage | Avg distinct docs@3 | Avg latency | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dense_multilingual_embedding` | 0.80 | 0.87 | 0.96 | 0.92 | 3.00 | 138.30 ms warmed validation run | current production winner |
| `lightrag_relation_graph` | 0.80 | 0.93 | 0.92 | 0.92 | 3.00 | 3.46 ms warmed validation run | experimental |
| `GraphRAG` | pending | pending | pending | pending | pending | pending | not implemented |

Latency note: the comparison uses warm retrieval latency in the same local
evaluation environment. Dense timing excludes the initial one-time embedding
model load after warm-up, but includes fixture loading, chunking, embedding, and
ranking for each case. LightRAG-style timing includes fixture loading, chunking,
in-memory graph construction, graph expansion, scoring, and final evidence
selection for each case. Earlier warmed comparison runs measured dense at
457.00 ms and LightRAG-style at 10.60 ms, so these small-corpus latency values
should be treated as local timing observations rather than deployment guarantees.

Case-level comparison:

- `climate_food_migration_chain`: LightRAG tied dense on coverage and retrieved the required climate and food-security documents.
- `displacement_health_service_chain`: LightRAG tied dense.
- `funding_health_food_security`: LightRAG tied dense.
- `regional_cooperation_support_chain`: LightRAG tied dense.
- `full_humanitarian_speech_evidence`: LightRAG improved required-source coverage from 0.33 to 0.67 by retrieving `health_displacement.txt`, but it still partially failed and did not recover `climate_agriculture.txt`. Relation-term hit rate dropped from 0.80 to 0.60 on this case.

Interpretation:

- LightRAG-style retrieval improved source coverage on the broadest synthesis case.
- LightRAG-style retrieval did not beat dense overall because relation-term hit rate was lower.
- Dense remains the current production winner.
- The LightRAG-style result is promising enough to justify further relation-graph experiments, but not strong enough to replace dense production retrieval.
- GraphRAG is still not implemented or measured.

## Future Comparison Placeholder

| Method | Phase 1 general retrieval | Phase 2 relation retrieval | Required-source coverage | Relation-chain coverage | Latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Sparse / RAG-lite | measured | not yet tested on Phase 2 | pending | pending | measured | completed |
| BM25 | measured | not yet tested on Phase 2 | pending | pending | measured | completed |
| Dense multilingual | measured | measured | 0.87 | 0.92 | 1736.45 ms original Phase 2 run; 138.30-457.00 ms warmed comparison runs | current production winner |
| Hybrid BM25 + dense | measured | not yet tested on Phase 2 | pending | pending | measured for Phase 1 | completed Phase 1 only |
| LightRAG-style local graph | not tested on Phase 1 | measured | 0.93 | 0.92 | 3.46-10.60 ms on warmed Phase 2 runs | experimental offline prototype |
| GraphRAG | not implemented | pending | pending | pending | pending | future experiment |

## What Not To Claim

- Do not claim BM25 improved retrieval quality.
- Do not claim hybrid beats dense.
- Do not claim dense solved semantic paraphrase.
- Do not claim dense is the final winner across all task types before Phase 2 graph comparisons are completed.
- Do not claim the LightRAG-style prototype beats dense overall.
- Do not claim the official LightRAG package was implemented.
- Do not claim GraphRAG or the official LightRAG framework was implemented.

## Next Recommended Work

- Add Phase 2 sparse/BM25 baselines only if the team wants a broader relation benchmark comparison.
- Test a stronger multilingual embedding model if semantic paraphrase remains weak.
- Compare LightRAG and GraphRAG later on the same Phase 2 cases.
- Integrate no graph method into production until it shows measurable value over dense retrieval on relation-heavy speech-generation tasks.
