# Phase 3 RAG Speech-Generation Evaluation Plan

## Objective

Phase 3 is designed to compare final generated speeches when the only changing
variable is retrieval. The target comparison methods are:

- `dense_multilingual_embedding`
- `lightrag_relation_graph`
- `graphrag_global_local`

This phase starts with an offline harness and deterministic mock generation.
No real LLM output is claimed yet.

## Fair-Comparison Controls

Every method must use the same:

- source documents;
- user request;
- target language;
- requested duration;
- canonical prompt template;
- LLM/provider in the later real-generation stage;
- temperature or deterministic generation setting;
- maximum output length;
- context budget.

The current harness fixes retrieval context at 3 evidence chunks and 3600
characters maximum. The prompt never exposes the retrieval method name to the
generator.

## Methods

- Dense multilingual retrieval remains the current production retrieval method.
- LightRAG-style retrieval is an offline relation-aware graph experiment.
- GraphRAG-style retrieval is an offline global/local graph experiment.

LightRAG-style and GraphRAG-style remain evaluator-only. They are not production
routers, endpoints, or product retrieval modes.

## Pilot Scope

The pilot uses 10 fixed cases:

1. `single_doc_exact_climate_en`
2. `single_doc_paraphrase_health_fr`
3. `single_doc_arabic_diplomacy_ar`
4. `numbers_entities_precision_en`
5. `multi_doc_direct_relation_food_health`
6. `multi_doc_chain_climate_migration_health`
7. `broad_synthesis_humanitarian_speech`
8. `distractor_documents_regional_funding`
9. `cross_language_request_fr_sources_en`
10. `arabic_output_from_multidoc_sources`

The cases use fictional local policy and humanitarian documents only. No
external downloads are required.

## Current Harness Status

Implemented:

- local fixture loading;
- fixed retrieval context budget;
- dense, LightRAG-style, and GraphRAG-style retrieval method slots;
- deterministic dense stub for environments without optional embedding runtime;
- canonical prompt builder;
- deterministic mock generator;
- structured result objects;
- retrieval/context proxy metrics;
- schema and pipeline tests.
- explicit real-generation evaluator mode for a future controlled Gemini pilot;
- preflight validation that checks configuration and run shape without calling a
  provider.

Not implemented yet:

- executed real Groq, Gemini, local Aya, or remote Aya generation;
- human speech-quality scoring;
- production endpoint integration;
- frontend integration;
- public retrieval-method selection.

## Proxy Metrics

The initial harness measures context availability only:

- evidence source coverage;
- key factual claim availability in retrieved context;
- expected relation-chain evidence availability;
- fixed context budget compliance;
- prompt-template equality except for evidence;
- source traceability map;
- output schema validity.

These proxy metrics are not final speech-quality metrics.

## Real Pilot Readiness

Mock structural validation is complete. Real-mode evaluator support is
implemented but has not been executed against a live provider.

Gemini is the proposed first provider once `GOOGLE_AI_KEY` is configured
locally. The initial real pilot is designed to use:

- 13 cases, including 10 original Phase 3 cases and 3 hard discriminative variants;
- 3 retrieval methods;
- 1 generation per method/case;
- 39 total generations;
- temperature 0;
- max_tokens 2800;
- fixed canonical prompt template;
- fixed 3-chunk / 3600-character context budget.

The real pilot must keep the provider, model, temperature, maximum output
length, prompt template, language, requested duration, and context budget fixed
for every retrieval method. The only intended variable is the retrieved
evidence.

Before running a real pilot, the evaluator can list Gemini text-generation
models with `--list-available-models`. A pilot may then use `--model <model_id>`
to select one discovered model for that evaluator process only. The override is
not written to `.env` and does not change production provider logic.

Dense retrieval is explicit in the evaluator: `--dense-mode stub` keeps the
fast deterministic stand-in, while `--dense-mode real` uses the local dense
multilingual embedding retriever and fails rather than silently falling back if
the optional model/runtime is unavailable.

The hard discriminative variants are benchmark-only additions that preserve the
original Phase 3 cases unchanged. They add plausible policy, funding,
coordination, multilingual, or cross-language distractors so each retrieval
method must make tradeoffs under the existing 3-chunk context budget:

- `multi_doc_chain_climate_migration_health_hard`;
- `cross_language_request_fr_sources_en_hard`;
- `arabic_output_from_multidoc_sources_hard`.

Hosted model output may still have minor non-determinism despite temperature 0.
No final speech-quality conclusion is permitted until real outputs are
generated and reviewed.

## Provider Completion Metadata

The first real Gemini outputs did not preserve provider completion metadata, so
incomplete outputs could not be confidently attributed to retrieval quality,
provider truncation, safety behavior, or another interruption. Future real-mode
evaluator runs now request normalized provider metadata, including finish
reason, token usage, safety ratings, candidate count, and whether provider
metadata was available.

This metadata is recorded only in ignored evaluation JSON reports. It is needed
to distinguish retrieval effects from provider-side completion behavior. No
quality conclusion about graph-style speech generation should be drawn from
incomplete outputs until completion metadata is available for comparable runs.

Follow-up metadata reruns showed that the graph-style broad-synthesis outputs
ended with Gemini `MAX_TOKENS`, with most of the configured output budget spent
on provider thought tokens rather than visible speech text. Future diagnostic
runs should use an explicit low Gemini thinking budget to reserve output
capacity and isolate retrieval quality from thinking-budget exhaustion.

The earlier truncated LightRAG-style and GraphRAG-style speeches should not be
used as evidence of graph-style generation quality until they are rerun with
completion metadata and a controlled thinking budget.

## Final Real-Generation Findings

### Evaluation Controls

The controlled real-generation comparisons used:

- provider: `gemini`;
- model: `gemini-2.5-flash`;
- temperature: `0`;
- max tokens: `2800`;
- thinking budget: `256`;
- real Dense mode for Dense comparisons;
- fixed canonical prompt template;
- fixed 3-chunk evidence context budget;
- blinded qualitative scoring;
- captured provider completion metadata.

### Why Thinking Budget Was Added

Earlier graph-style outputs for the broad-synthesis case ended mid-sentence
with provider finish reason `MAX_TOKENS`. Provider metadata showed that Gemini
thought-token usage consumed most of the configured output allowance. With
`thinking_budget=256`, all fair comparison outputs completed with finish reason
`STOP`. The truncated pre-control outputs should not be used to judge retrieval
quality.

### Discriminative Case Results

#### `broad_synthesis_humanitarian_speech`

Dense real retrieval had lower expected-source coverage because it missed the
food-security and migration evidence. LightRAG-style produced the strongest
blinded speech score, `34/35`, followed by GraphRAG-style at `28/35` and Dense
real at `27/35`.

Interpretation: LightRAG-style improved broad synthesis by retrieving the
food-security -> migration -> health evidence chain.

#### `arabic_output_from_multidoc_sources_hard`

The blinded Arabic scores were:

- Dense: `33/40`;
- LightRAG-style: `34/40`;
- GraphRAG-style: `39/40`.

Interpretation: GraphRAG-style achieved complete evidence coverage and produced
the strongest Arabic multilingual synthesis.

#### `cross_language_request_fr_sources_en_hard`

The blinded French scores were:

- Dense: `39/40`;
- GraphRAG-style: `33/40`.

Interpretation: Dense preserved the complete climate-finance and
regional-funding evidence bundle. GraphRAG-style selected a distractor and
missed the regional-funding source.

### Final Comparison Table

| Case | Dense | LightRAG-style | GraphRAG-style | Best method | Main reason |
| --- | ---: | ---: | ---: | --- | --- |
| `broad_synthesis_humanitarian_speech` | `27/35` | `34/35` | `28/35` | LightRAG-style | Retrieved the food-security -> migration -> health evidence chain that Dense missed. |
| `arabic_output_from_multidoc_sources_hard` | `33/40` | `34/40` | `39/40` | GraphRAG-style | Achieved complete expected-source, key-claim, and relation-chain coverage for the multilingual Arabic synthesis. |
| `cross_language_request_fr_sources_en_hard` | `39/40` | Not measured | `33/40` | Dense | Preserved both climate-finance and regional-funding evidence; GraphRAG-style selected a distractor. |

### Production Recommendation

Keep Dense as the current production default. Do not implement automatic
routing yet. LightRAG-style and GraphRAG-style are validated research
prototypes, not production replacements.

Graph-aware retrieval appears beneficial for some broad multi-document and
multilingual relation-synthesis tasks. Dense remains strong for focused
retrieval and French cross-language semantic retrieval. No universal winner was
found, and more cases are needed before routing or product changes.

### Latency Caveat

Real Dense latency measurements may include cold model initialization. Do not
compare current Dense retrieval latency directly with graph-prototype retrieval
latency until warm Dense timing is measured. Generation latency dominates total
end-to-end latency in the Gemini evaluations.

### Research Limitations

Only three discriminative real-generation cases were fully evaluated. The graph
prototypes are local deterministic benchmark implementations, not the official
LightRAG or Microsoft GraphRAG packages. Qualitative scores are
benchmark-based and should be complemented by additional human and
native-language review. Generated JSON remains ignored and was not committed.

## Future Human Scoring Rubric

Later real-generation runs should use 1-5 scoring for:

- grounding;
- hallucination avoidance;
- topic completeness;
- multi-document completeness;
- coherence;
- interpreter suitability;
- language quality;
- evidence traceability.

## Explicit Caution

Mock generation validates only the evaluation pipeline structure. It does not
prove fluency, language quality, interpreter suitability, hallucination rate, or
final end-to-end superiority. Those claims require real generation under fixed
provider settings plus human review.
