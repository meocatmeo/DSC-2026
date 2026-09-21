# Depth-100 implementation

> **Outcome update, 21 September 2026:** the smaller 256/512-question pilots
> produced development gains but private regressions (0.9464/0.9459 vs 0.9510).
> See [the developer result ledger](../development/EXPERIMENT-RESULTS.md).
> This document describes the original implementation and proposed full run;
> it is not evidence that the full 1,569-question run was executed or improved.

Implemented as an isolated experiment in `legalir2/deep_pool.py` and
`legalir2/deep_fusion.py`. Launch instructions are in
[the 4090 runbook](../runbooks/RUNBOOK-deep-pool-4090.md).

The frozen inference modules, model weights, previous releases and scored
0.9510 private answers are unchanged. No GPU inference or new leaderboard score
has been measured locally.

## Implemented behavior

- Actual fused top-100 retrieval plus every original duo candidate; no top-eight
  truncation. Gold labels do not participate in candidate construction.
- Existing BF16 G32 and Qwen rerankers score fulltext-v2 evidence. Raw scores,
  channel rankings, passage text and token IDs are retained for reproducibility.
- Twenty-three linear coefficients trained on 1,569 frozen training questions,
  using fixed 300 epochs and L2=0.001. Development/validation IDs and normalized
  aliases are excluded. Missing V9 scores have an explicit presence feature.
- At most four additional comparator candidates, chosen by learned fusion.
  Every original candidate, passage and pairwise comparison is preserved.
  Both A/B orders are scored only for edges involving new documents.
- Development selects baseline or the expanded tournament. Ties retain the
  baseline. A positive fixed-policy validation difference is required before
  private scoring. Full-text validation baseline is collected when needed.
- Candidate recall is reported at fused top 50, fused top 100, the scored pool,
  original duo pool and expanded duo pool, separately from final Recall@5.
- Submission packages both V9 and duo. V9 is unchanged; duo falls back to the
  exact scored answers when development or validation rejects the new policy.

The wider pool is normalized consistently in training and inference. The saved
V9 scores remain an explicit feature rather than being reinterpreted as a
depth-100 stacker. Fusion fitting does not update neural model weights.

## Verification

The new algorithm/launcher tests and existing baseline/auto-stop checks pass.
Tests cover preserving candidates beyond the cutoff, raw-feature replay,
query-to-query evidence reset, normalized-duplicate leakage rejection,
label-independent admission, unchanged original comparisons, both comparison
orders, baseline selection on ties, validation rejection and actual ZIP output.

Real source checks verify 1,569 training records, 429 development comparisons,
all 2,080 scored private predictions and the released dataset/hash chain.
Two real corpus questions exercise 202 full-text pointwise bundles and 296
new-pair sequences: maximum observed lengths were 1,187 and 2,031 tokens
respectively. Model forwards and retrieval were mocked for this local smoke
test; the corpus, tokenizer, passage selection and feature checks were real.

The conservative combined parameter audit is 2,327,063,576, including the 23
new fusion coefficients, below the 4,000,000,000 limit. This is a parameter
audit, not a blanket interpretation of competition rules.

## Release and operations

Release: `20260920-deep-pool-4090-v1`.
Run: `private-deep-pool-4090-v1`.
The launcher follows the existing setup/restore/lock/log/backup structure,
adds training-score collection and CPU fitting, and retains submit and Vast
auto-stop after verified successful backup. All shell files remain gitignored.
Only DSC cloud paths are used.

Published to `gdrive:DSC-2026-results/releases/private/20260920-deep-pool-4090-v1`.
ZIP SHA-256: `3f54c78515e40c861baa8a71e2654651563335ba9ddffd9ef9590881df15e82a`.
Downloaded cloud verification found two matching files and zero differences.
A local publication marker prevents overwriting this release.

This job is larger than focused rescue. There are 1,569 training questions
before the 429-question development gate. Read the runbook's timing scenario
and use the measured first-shard throughput before budgeting a complete run.
An illustrative 10-second collection and 8-second comparison rate yields about
6.5 hours through development or 25 hours for the complete path, excluding
setup, CPU fitting and transfers. These are not GPU benchmarks.

No new private score or guaranteed improvement is claimed. The broader
candidate-coverage result motivated the experiment; it did not establish that
the learned ranker can recover those answers without other losses.
