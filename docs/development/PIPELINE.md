# Pipeline and implementation notes

See [current status](README.md) before running an experiment. The newest model
is not the best measured private model.

## Retrieval and the scored full-text baseline

The corpus has **8,532 document IDs**. `legalir2/pipeline.py` retrieves article
chunks through lexical BM25F/bigram and dense embedding channels, groups them
by document, and combines rankings with weighted reciprocal-rank fusion. The
frozen V9 route scores a 50-document candidate pool and uses its saved fusion.

The trained duo comparator evaluates pairs of candidate documents in both A/B
orders. Pair probabilities feed a regularized tournament fit. At strength 0.25,
the final score is `prior + 0.25 * (ability - prior)`. Five document IDs are returned.
See `legalir2/duo.py` and `legalir2/private_infer.py` for the frozen path.

The successful full-text variant uses `legalir2/duo_evidence.py`:

- A corpus-wide SQLite FTS index contains 220-word windows with 64-word overlap.
- Full-text expansion proposes at most two extra documents. It requires a
  matching phrase of at least six tokens and query-term coverage of at least 0.55.
- Within each candidate, query-matching passages are selected, deduplicated,
  and packed under the actual tokenizer budget. Labels consume part of the budget.
- The pair context is bounded to 2,048 tokens, including the query and prompt.
  Evidence traces retain document IDs, passage text, token IDs and hashes.

In the saved 0.9510 private-run audit, none of the extra phrase candidates entered
the submitted five. This limits explanations of the gain: extra candidate
availability alone does not explain it. Passage changes and their ranking
effects need to be distinguished from retrieval expansion.

## Deep-pool experiment

`legalir2/deep_pool.py` and `legalir2/deep_fusion.py` implement the unsuccessful
private deep-fusion experiments, which remain useful for reproducible research.

1. Collect the real fused **top 100**, retaining every original duo candidate.
2. Select full-text evidence; reuse the original duo passage traces exactly for
   its retained candidates. Reset `Evidence.trace` for every query: `encode`
   appends to the trace, while `expand` historically performed the reset.
3. G32 scores selected passages at its saved 768-token limit. The Qwen pointwise
   reranker scores a document bundle of up to three passages at 2,048 tokens.
4. Save raw scores, channel ranks, evidence and the feature matrix. The 20
   version-2 stack features are extended with V9 score, reciprocal rank and a
   presence indicator, giving **23 features**.
5. Fit a linear model using multi-positive listwise targets, fixed 300 epochs,
   and L2=0.001. Candidate selection never uses held-out gold labels.
6. Append up to **four** learned candidates to the original duo pool. Normalize
   the learned scores over that shortlist to form the new prior.
7. Preserve original pair probabilities and evaluate both orders for new edges.
   Fit the enlarged tournament using `fulltext_prior.fit` and return the top five.

Preserving the original candidate pool does **not** preserve its top-five
answers. The learned prior reranks the whole pool. This explains how the method
can change many answer sets despite adding only four candidates.

### Missing ranks and normalization

`stack.build_features(..., version=2)` gives an absent channel document zero
reciprocal-rank credit. Version 1 preserves the historical `len(candidates)+1`
sentinel solely for replaying old weights. Both rerankers contribute features.

Feature normalization depends on the candidate pool. Changing a 50-document
pool to 100 and keeping the same fitted weights is not an equivalent model.
Train and evaluate with consistent normalization; preserve the raw scores so
the feature matrix can be checked independently.

Historical source comments claiming that a small linear model cannot overfit
are not an evaluation guarantee. Candidate rows from one question are not
independent labeled questions. These experiments did regress privately.

## Splits and training sizes

| Data | Size | Role |
| --- | ---: | --- |
| Original labeled training file | 7,000 | Contains records assigned to multiple historical roles; do not use all blindly |
| Frozen question memory | 5,518 | Excludes held-out IDs and normalized question aliases |
| Cached V9 calibration train | 1,998 | Includes the 429 development questions |
| Eligible deep-fusion training subset | 1,569 | Cached training questions remaining after held-out exclusions |
| Pilot training | 256, then 512 | Nested deterministic sample of that eligible subset |
| Development | 429 | Policy selection and repeated diagnostics |
| Validation | 1,050 | Separate labeled set, also historically reused |
| Private | 2,080 | Hidden labels; inference only |

The pilot sample sorts salted SHA256 values of normalized question text, with
ID as tie-breaker. The 512 sample retains the exact original 256 as a prefix.
Sampling does not use labels, scores or development errors. Training-stage GPU
work collects scores; fitting the 23 coefficients runs on CPU.

Private input keeps `answer: null` on disk. `private.question_records` converts
it to `[]` in memory because scoring helpers expect an iterable. Diagnostic
scripts must use the same loader; a direct JSON load can fail identity checks.

## Module map

| Module | Responsibility |
| --- | --- |
| `deep_pool.py` | Guarded raw collection, fitting, comparison, selection, validation, submission |
| `deep_fusion.py` | Features, learned scores, shortlist and final ranking |
| `deep_pilot.py` | Fixed sample; development-only finish with baseline ZIPs |
| `deep_pilot512.py` | Nested extension and verified reuse of 256 train + 429 development raw scores |
| `deep_pilot_submit.py` | Exact 256-model private continuation, validation skipped |
| `deep_pilot512_submit.py` | Exact 512-model private continuation and compatible private raw-cache import |
| `fulltext_prior.py`, `consensus.py` | Earlier prior/consensus experiment and tournament fitting |
| `focused_retrieval.py`, `related_documents.py`, `retrieval_rescue.py` | Earlier focused/citation rescue experiments |

The teammate integration is documented in
[THONG-INTEGRATION.md](../state/THONG-INTEGRATION.md). Consensus ideas were ported
with ID alignment and split guards. All-label memory/filtering was excluded.
No quantization implementation was found in the audited teammate branches.

## Coding and compatibility conventions

Use small functions, plain dictionaries, `Path`, NumPy and an `argparse` stage
entry point, matching nearby modules. Use `read_json`, `write_json`, `digest`
and `guard` for persisted state. Validate IDs, ordering, shapes, finite values,
source hashes and evidence before accepting a cached shard. Preserve deterministic
document-ID tie-breaking. Never silently rebuild or overwrite mismatched caches.

Frozen modules, model bytes, indexes and release manifests are hash-bound.
Implement a new experiment in an additive module/release with a separate run ID.
Do not edit code in a running GPU workspace to make a signature check pass.
The later modules follow this rule even where it means some deliberate duplication.

The audited combined model count is **2,327,063,576 parameters**, conservatively
including the 23 coefficients. All four neural checkpoints remain unchanged in
the 256/512 pilots. This count is not a blanket competition-rule certification.
