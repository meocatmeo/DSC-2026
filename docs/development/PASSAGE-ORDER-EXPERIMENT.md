# Passage relevance ordering — isolated experiment

22 September 2026. **Implemented and CPU prepared; no new GPU predictions or
measured score gain yet. The frozen 0.9523 release is unchanged.**

## Hypothesis and scope

The existing pointwise passage reranker can identify a relevant clause that the
lexical evidence sorter places behind an unrelated passage. Test whether putting
the more relevant passage first helps duo use that clause.

The first arm changes **ordering only**: sort the already packed passages by the
existing first reranker's score, descending with stable ties. It does not retrieve
new passages, drop passages, allocate a different token budget, retrain weights,
alter document candidates, or change priors. Those would be separate experiments.
In particular, this arm cannot recover gold documents outside the existing pool.

Recompute both the original and retrained comparator using the reordered evidence;
apply the identical current agreement policy to those two results. Compare with
the current agreement answers. The policy is fixed, but its first four answers
can move when the underlying comparator scores change. This experiment does not
claim to preserve the previous answer sets.

The 429 development questions have already been inspected and reused. The result
will be diagnostic, not an independent holdout. No validation or private inference
is included, and the module has no submission stage.

## Prepared inputs

`reports/passage-order-v2/` contains `plan.json`, `prepared.json`, and 429 question
files. The earlier `passage-order-v1` was a preparation smoke check; v2 adds runtime
matching and unchanged-input replay checks and is the runnable experiment.

The preparation verifies:

- Full baseline/old-evidence/retrained cache digest against the saved screen.
- All deep comparison rows against their saved calibration digest, and each raw
  passage-score row against the corresponding comparison's digest.
- Exact original-pool evidence equality between the baseline and deep raw cache.
- Both checkpoint inventories and identical tokenizers.
- Reconstruction of every saved input token from its label and packed passages
  before sorting. A mismatch fails; no approximate retokenization is accepted.
- Same passages and total token count after ordering, unchanged pool and prior.

Preparation finds 7,464 document instances, of which 5,447 change order; all 429
questions have at least one change. Each comparator needs 56,769 unordered pairs
rescored, both orientations: 227,076 total directed predictions across two models.
It reuses 4,596 pairs per model (7.5%). This is therefore not a cheap partial rescore.

## GPU evaluation

Requires local original-duo and retrained-control checkpoints, a BF16-capable CUDA
GPU, FlashAttention 2.8.3, Torch 2.7.1+cu128, Transformers 4.51.3, tokenizers 0.21.4,
NumPy 2.2.6 and SciPy 1.15.3. No quantization. The frozen `plan.json` checks versions
and model hashes. Only one comparator is resident at a time, suitable for testing
on the 24 GB 4090 used previously. Actual GPU memory and execution are untested.

From the repository root, after installing the matching environment and copying
the prepared directory and the two checkpoint folders:

```bash
python -m legalir2.passage_order_run score \
  --out reports/passage-order-v2 --models /path/to/model-vault --batch 4 \
  --max-questions 2

python -m legalir2.passage_order_run score \
  --out reports/passage-order-v2 --models /path/to/model-vault --batch 4

python -m legalir2.passage_order_run evaluate --out reports/passage-order-v2
```

The first command smoke-tests two questions per model. Resume with the second
command; previously committed question files are verified and reused. Evaluation
requires all 429 questions for both models. It reports recall difference, wins,
losses, changed sets, paired bootstrap interval and sign test, with no promotion.

Each newly loaded model first reruns the first unmodified question. Its original
top-five prediction must match exactly and maximum probability deviation must be
at most 0.0001. A failure stops scoring; investigate runtime/model differences
rather than relaxing the tolerance to get the run through. This is a one-question
compatibility check, not a complete baseline rerun.

Old comparisons are reused only when both document input token sequences are
unchanged. Both orientations of each changed pair are measured. Checkpoints are
written after every question. Inputs, source code and completed score files have
identity checks. Preserve source bytes when copying: normalizing line endings
would change the prepared code hashes.

Based on the earlier roughly 8.1 seconds/question/comparator logs, a rough planning
estimate is about 1.5–2.5 hours for both development arms, excluding setup/download.
This is not a benchmark of the new runner; use the smoke run to revise it before
committing to the full run. No GPU rental or cloud upload has been initiated.

## CPU checks and limitations

Targeted tests cover token preservation, stable ties, corrupted traces, incident
edge selection, bidirectional probability averaging, exact reuse of other edges,
cache identity, prior protection and fitted-score integrity. The prepared inputs
do not need corpus/index files or a passage reranker at GPU scoring time because
all original passage scores and token IDs are already frozen.

All 13 targeted experiment/agreement tests passed. Reloading all 429 prepared
files reproduces the current agreement development recall, 0.9526029526, exactly
to the reported precision. The best private ZIP still matches its frozen SHA-256.
See `reports/passage-order-v2/cpu-verification.json`. Seven gold documents remain
outside the current answers but inside the pool; this policy changes the first
passage for three of them. That is an input-order observation, not a recovered
answer or a measured gain.

The reused pointwise model has seen some historical training data; this experiment
does not establish model-training independence. The motivating cases were selected
after errors were inspected. No private labels were used, and no manual document
ID exceptions are part of the policy. A new score increase is not guaranteed.
