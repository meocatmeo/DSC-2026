# Developer handoff: private LegalIR experiments

Updated 22 September 2026. Start here before modifying or running the model.

## Current decision

**Keep the CPU agreement submission scoring 0.9523 as the best private reference.**
The user reported this score on 22 September for the fixed fifth-slot agreement
rule. It improves on 0.9510 by 0.0013 (0.13 percentage points), while preserving
the previous baseline's first four answers for every question. The 0.9510 ZIP
remains the frozen algorithmic anchor; do not recursively apply the replacement
rule to the new submission.

New best ZIP: `submissions/private/20260922-duo-agreement-v1/submission-private-agreement.zip`.
SHA256: `37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f`.

The newer deep-fusion submissions scored **0.9464 with 256 training questions**
and **0.9459 with 512**. Their development gains did not generalize. Neither is
a replacement for the best submission, even though their files and execution
passed integrity checks.

The subsequent CPU FTS threshold sweep also failed its prerequisite: best
pooled candidate coverage improved by +0.003043, below the required +0.008.
See the coverage note below. Conditional admission scoring remains inactive.

The retrained full-text comparator scored **0.9500** privately. Its CPU audit
reproduces all 2,080 answers and finds 719 changed sets, with no cache-integrity
failure. On the existing development split, 129 of 130 changed sets leave recall
unchanged; 24 of 31 remaining gold-document misses are outside the comparator
pool. See the new audit before scheduling more retraining.

These private scores were reported by the user. Private gold labels are not
available locally. The user wrote "216" in the final score message; we interpret
that as 256 because the preceding uploaded file was explicitly identified as
the 256 model. Preserve that qualification until corrected or confirmed.

| Read next | What it explains |
| --- | --- |
| [Pipeline and coding conventions](PIPELINE.md) | Retrieval, passages, duo, fusion, splits, and code ownership |
| [Experiment results and lessons](EXPERIMENT-RESULTS.md) | Measured scores, answer changes, failure interpretation, and next experiments |
| [Retrained duo CPU audit](RETRAINED-DUO-CPU-AUDIT.md) | 0.9500 replay, eight-case evidence review, and candidate/prior failure decomposition |
| [CPU agreement experiment](DUO-AGREEMENT-CPU.md) | One-slot replacement with two-model support; reported private score 0.9523, current best |
| [Frozen 0.9523 Docker/source handoff](../runbooks/RUNBOOK-best-09523-handoff.md) | Exact offline Docker recovery; models/source preserved; fresh GPU inference and organizer acceptance still unverified |
| [Deadline agreement alternatives](DEADLINE-AGREEMENT.md) | Three CPU-only speculative submissions; four attempts remaining, no new private gain established |
| [Supporting-authority candidates](LOCAL-AUTHORITY-CANDIDATES.md) | CPU citation/name retrieval and coverage diagnostics; experimental, no answer changes |
| [CPU FTS coverage and score gate](FTS-COVERAGE-GATE.md) | 72 policies tested; +0.003043 pooled coverage fails the +0.008 prerequisite |
| [Current fusion and fourth voter](CURRENT-FUSION.md) | Exact old-policy check fails development; current full-ranking fusion gains one question; optional ViRanker scorer |
| [Duo full-text training rebuild](DUO-FULLTEXT-TRAINING.md) | Old evidence audit, empty-source quarantine, isolated control and hard-negative training datasets |
| [Artifacts and GPU operations](ARTIFACTS-AND-OPERATIONS.md) | Exact submissions, cloud runs, releases, cache reuse, recovery, and auto-stop |
| [Private-test state](../state/PRIVATE-TEST.md) | Longer history of the private target |
| [Project layout](../PROJECT-LAYOUT.md) | Repository and DSC cloud organization |

## What a new teammate should know first

- The active input is `data/private-official.json`: **2,080 questions** with
  hidden answers. Public scores such as **0.9508** belong to a different test.
- Each output contains five distinct corpus document IDs per question. This
  task retrieves documents; it does not generate legal answers in prose.
- Development has **429 questions**; the separate labeled validation set has
  **1,050**. Both have been used historically. Do not call either a fresh test.
- The 256/512 experiments train **23 linear fusion coefficients**, not the
  embedding, pointwise rerankers, or neural duo comparator.
- More candidate gold coverage is an opportunity, not a measured final score.
- The 256 and 512 pilots stop after development and produce **unchanged baseline
  ZIPs** for the existing backup workflow. Separate submission launchers perform
  actual private inference. The word `submit` alone does not identify a new model.
- The direct private continuations intentionally **skipped the 1,050-question
  validation**, at the user's request to control cost. Their reports record this.
- Current experiments use **BF16, no quantization**. The regressions are not
  evidence that quantization hurt the results; these runs were not quantized.

## Original 0.9510 anchor to preserve

Run: `private-duo-fulltext-v2-4090-fast1`.

Local submission:
`submissions/private/20260920-fulltext-v2-09510/duo/submission-private-duo.zip`.

SHA256:
`40caa611cfdbf4f853296fdfef93774b117fffc4c339d591690080a59e0a6d57`.

Cloud copy:
`gdrive:DSC-2026-results/runs/private/private-duo-fulltext-v2-4090-fast1/artifacts/submission-private-duo.zip`.

## Recommended next work

Start with a CPU audit of saved labeled scores. Measure which correct documents
are lost at candidate admission and final ranking, then test a small, fixed
change against the 0.9510 method. Report wins, losses and uncertainty as well as
mean recall. No new improvement is established by this handoff.

Do not start another full GPU run merely because a development mean is slightly
higher. The next experiment should have an explicit hypothesis, compute budget,
and a credible evaluation plan. Keep private leaderboard outcomes as aggregate
results; they do not reveal question-level gold labels.

## Sharing this work

These notes are local Markdown files; writing them does not commit or push the
working tree. There are existing uncommitted source changes. Review them before
sharing a Git revision. All `.sh` files are intentionally gitignored; generated
models, reports, caches, submissions and releases are also ignored. A Git clone
alone is therefore not a runnable GPU workspace. Obtain the required immutable
release and authorized rclone configuration separately. Never share credentials.
