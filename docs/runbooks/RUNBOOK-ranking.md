# Missing ranks and document-local passages

The default passage selector now ranks chunks **inside each candidate document**,
using the full lexical and dense scores already computed for the query. It fuses
those local ranks with the configured channel weights and RRF constant, then picks
distinct passages. It can reach articles below the global chunk cutoff and scans
past duplicate windows until enough distinct passages are found. Candidate document
rankings, passage count and reranker token limits are unchanged.

Set `"passage_selection": "retrieval"` in `--settings` to reproduce the old global
chunk-prior selector. Set `"passage_selection": "document"` explicitly in new run
configs. The selected mode and its time are recorded in the run manifest. V11's
frozen-baseline collector uses `retrieval` when the old manifest has no selector.

## Stacker feature versions

New stacker features use `feature_version: 2`: a document absent from a channel
gets zero reciprocal-rank credit, below every real channel rank. The feature count
and order are unchanged. The lexical/dense rank-gap feature uses the corrected
reciprocal ranks too.

Existing unversioned stacker files and feature caches are treated as version 1 so
their coefficients retain the old meaning. Training on an old feature cache still
produces version 1 weights. New collection and fitting produce version 2 weights;
mixed-version caches fail. The raw-score `cv --fit-stacker` path rebuilds corrected
features and saves version 2 weights.

**Use a new feature cache and refit the stacker to activate the correction.** Do not
add a version field to an old weight file: that would change the inputs without
refitting its coefficients. Do not reuse old passage scores for the new selector.
No dense-index rebuild or base-model retraining is needed for either code change.

## Controlled comparison

Use the original corpus/index and the exact V9 model checkpoints. First replay V9
with its old stacker and `passage_selection=retrieval`. Then collect and fit a new
stacker with the same evidence mode to isolate the missing-rank correction. Finally
collect a separate cache with `passage_selection=document` and refit. Evaluate all
three on the same validation questions; retain the V9 submission until a comparison
justifies promotion. Fitting on the same calibration questions as V9 isolates a
change but does not resolve their overlap with reranker training.

Example for the **combined** change on the Linux GPU workspace, from the repository
root. These are new artifact names; `stack` otherwise reuses an existing cache.
Confirm the dense index and both reranker directories exist before running.

```bash
.venv/bin/python -m legalir2 stack \
  --reranker-model models2/reranker-g32 models2/partner_qwen \
  --questions artifacts/splits/train.json --limit 2000 \
  --settings '{"lexical_weight":0.75,"dense_weight":1.0,"candidates":50,"stage2_candidates":50,"passages":3,"rerank_max_length":768,"passage_selection":"document"}' \
  --cache artifacts2/groups/stack-document-v2.jsonl \
  --output outputs/stack-document-v2.json

.venv/bin/python -m legalir2 predict \
  --reranker-model models2/reranker-g32 models2/partner_qwen \
  --questions artifacts/splits/validation.json --evaluate \
  --settings '{"lexical_weight":0.75,"dense_weight":1.0,"candidates":50,"stage2_candidates":50,"passages":3,"rerank_max_length":768,"passage_selection":"document","stacker":"outputs/stack-document-v2.json"}' \
  --output outputs/validation-document-v2.json

.venv/bin/python -m legalir2 compare \
  --before outputs/v9-ensemble.diagnostics.json \
  --after outputs/validation-document-v2.diagnostics.json \
  --label "corrected missing ranks and document-local passages"
```

For the missing-rank-only arm, change the selector to `retrieval` in both commands
and use separate `stack-ranks-v2` / `validation-ranks-v2` filenames. Compare candidate
recall, final Recall@5 and per-question wins/losses. The code and regression tests
establish the new behavior; a leaderboard improvement has not been measured.

## Tests

```bash
python -m unittest tests2.test_ranking_evidence -v
python -m unittest discover -s tests2 -v
```

The focused tests cover deep real ranks versus absence, old-weight replay, feature
version propagation, both rerankers seeing recovered evidence, unchanged candidate
rankings and token limits, one score computation per query/channel, different dense
and lexical index orders, lexical-only and empty-query fallback, and duplicate
passages beyond the old pool limit.
