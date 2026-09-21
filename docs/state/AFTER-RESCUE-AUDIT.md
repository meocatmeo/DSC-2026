# After the failed focused-rescue experiment

Follow-up: [depth-100 implementation and verification](DEEP-POOL-4090.md).

The next experiment should address candidate truncation before changing the
final answer policy again. The private reference remains the user-reported
0.9510. No new private answers, release, GPU job or model update were made by
this audit; all models remain unquantized BF16.

## Replayed evidence

Downloaded the completed 429-question development cache from
`gdrive:DSC-2026-results/runs/private/private-retrieval-rescue-4090-v1/artifacts`.
Verified its release code, asset identities, complete shard membership,
per-row baseline hashes, candidate plans, evidence token hashes, probability
matrices and exact calibration score digest. All five reported recalls replay.

| Replacement policy | Changed questions | Better | Worse | Recall |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0 | 0 | 0 | 0.950272 |
| Focused, threshold 0.75 | 215 | 0 | 5 | 0.941531 |
| Focused, threshold 0.90 | 145 | 0 | 3 | 0.945027 |
| Citations, threshold 0.75 | 237 | 1 | 5 | 0.942696 |
| Citations, threshold 0.90 | 155 | 1 | 3 | 0.946193 |

Most changed answers merely exchange two unlabelled candidates. Very confident
pairwise wins do not imply a document matches the benchmark answer. One error
asks about a loan that is explicitly **not** a credit contract (question 27654):
the focused comparator gives 0.9697 to a credit-contract guidance document over
the labelled Civil Code. With old passages it gave that same challenger 0.3475.
Showing more query words did not preserve the exclusion in this case.

Another case asks about Thanh Hoa (22292), whereas the saved gold document's
title names Vinh Phuc. This is an apparent question/document-label mismatch,
not proof of a legal or annotation error. It argues against applying a hard
province filter without first checking the supplied corpus and labels.

There are 32 missed gold-document instances in the baseline. Twenty-four lie
outside its duo pool. The rescue run compares only six of the 32: its top-eight
restriction excludes five original-pool misses while citations add three new
ones. Consequently, 26 misses cannot possibly be repaired by any policy using
these scored candidates. Even an oracle choosing one permitted replacement
would reach only 0.959596 on this development split. That is diagnostic coverage,
not measured model accuracy or a private prediction.

Details: `reports/private/20260920-rescue-development/audit.json`.

## CPU alternatives rejected

Tested three fixed alternatives on the saved scores: require old/new view
agreement; or apply a bounded change in pairwise scores to the existing ranking
prior at strengths 0.25 and 0.50. Their development recalls were 0.947941,
0.946775 and 0.948524 respectively. None beat baseline.

A grouped five-fold selection probe also regressed: 0.947941 versus 0.950272.
The inner split uses separately salted SHA256 of normalized question text;
reusing the original CRC32 splitter is invalid because all development rows
were selected from its fold zero. Duplicate normalized questions stay together.
These are exploratory checks on historically reused development data, not a
fresh test. No policy is promoted and validation was not opened for these
failed ranking alternatives.

Details: `reports/private/20260920-rescue-development/cpu-alternatives.json`.

## Stronger direction: retrieve and score deeper candidates

The fused retrieval stage already knows more useful documents than the first
reranking stage sees. Nine development misses sit at ranks 51–100. They do not
require speculative citation expansion: the initial 50-document cutoff excludes
them before reranking.

Selected depth 100 from development and wrote the plan before opening the
saved validation rankings. The same fixed depth exposes ten additional gold
document instances on validation:

| Split | Questions | Candidate recall at 50 | At 100 | Newly reachable gold instances |
| --- | ---: | ---: | ---: | ---: |
| Development | 429 | 0.967754 | 0.984071 | 9 |
| Validation | 1,050 | 0.980476 | 0.985794 | 10 |

Validation was historically reused. These numbers measure retrieval
availability, **not a +0.0053 final-score gain**. Gold ranks are used only to
measure coverage and never to construct inference candidates.

The next implementation should:

1. Collect actual fused top-100 IDs and score the additional candidates with
   the existing rerankers, using the successful fulltext-v2 passages. Preserve
   all original duo candidates; do not apply the failed top-eight truncation.
2. Keep the 0.9510 baseline reproducible. Simply changing `candidates=100` is
   insufficient: stacker features are normalized within each candidate pool,
   so expanding the pool changes existing scores too. Record raw scores and
   channel ranks, then train/evaluate fusion consistently at the new depth.
3. Train candidate admission on frozen training questions, including difficult
   retrieved non-gold documents. Explicit exclusions, amendments, document
   scope and highly similar irrelevant statutes are useful error categories.
   Development/validation mistakes are diagnostics, not new training labels.
4. Evaluate candidate retention after each cutoff and Recall@5 on development
   before any complete private run. Broadly comparing all 100 documents with
   each other is unnecessary; rerank first, then compare a bounded shortlist.

This is the strongest next experiment supported by the current audit. Its
private score benefit and added GPU cost have not been measured. A later,
larger intervention is hard-negative fine-tuning of the existing embedding
model for remaining retrieval misses; that needs a new corpus index and a
separate validation, so it should follow the cheaper depth-100 experiment.

Reproduce with `scripts/analysis/audit_rescue_development.py`,
`scripts/analysis/test_rescue_alternatives.py`, and
`scripts/analysis/audit_candidate_depth.py`. The registered depth plan and
validation coverage are under `reports/private/20260920-rescue-development/depth/`.
