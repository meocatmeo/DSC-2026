# Artifacts, reproduction and GPU operations

Updated 21 September 2026. These instructions describe existing experiments,
including failed ones. See [results](EXPERIMENT-RESULTS.md) before launching.

## Exact private submissions

| Candidate | Run ID | Reported score |
| --- | --- | ---: |
| Best full-text baseline | `private-duo-fulltext-v2-4090-fast1` | **0.9510** |
| Deep 256 | `private-deep-pilot-submit-4090-v1` | 0.9464 |
| Deep 512 | `private-deep-pilot512-submit-4090-v1` | 0.9459 |

ZIP SHA256 values, in the same order:

```text
baseline  40caa611cfdbf4f853296fdfef93774b117fffc4c339d591690080a59e0a6d57
256       48f5bd2a6d363800a083874e52b3bf572c8ba91b8080f82205124bbc8ae57d15
512       352e4ae7bad4d5f469fa2f22c77d8500d17f04443dfc06316256da34b028c877
```

For each run, cloud artifacts are under
`gdrive:DSC-2026-results/runs/private/<run-id>/artifacts/`.
Use `submission-private-duo.zip`, its `.run.json` sidecar and
`submission-choice.json` together. A valid ZIP or a filename containing `duo`
does not establish which policy produced it. Check the hash and provenance.

The user uploaded 256 as `submission-private-duo (2).zip` and 512 as
`submission-private-duo (3).zip`. Those names are download suffixes, not versions.

## Dependencies and source artifacts

| Artifact | Location |
| --- | --- |
| Private questions | `data/private-official.json` |
| Frozen corpus store | Local `artifacts2/frozen-20260918/store.sqlite`; GPU `artifacts-ranking-h100/store.sqlite` |
| Shared embedding/G32/Qwen models | `gdrive:DSC-2026-results/assets/models/{embedding,reranker-g32,partner_qwen}` |
| Trained duo checkpoint/signature | `gdrive:DSC-2026-results/runs/duo/duo-20260918T110105Z/artifacts/` |
| Frozen source indexes | `gdrive:DSC-2026-results/runs/ranking-h100/20260918T013034Z/indexes/` |
| Original full-text private traces | Best baseline run's `artifacts/cache/{v9,duo}/` |

Private input SHA256:
`9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc`.
Frozen store SHA256:
`bf3799fe1af0ba58f61ba9861ff35dc6c9d352f2a6d7b822dbd7f63873cb5f7a`.

Restore exact frozen bytes. Do not substitute another local `artifacts2` store
or rebuild an index silently just because the document IDs look the same.

## Release and stage map

All release paths below are relative to
`gdrive:DSC-2026-results/releases/private/` and are immutable.

| Release | Archive | Purpose |
| --- | --- | --- |
| `20260920-deep-pool-4090-v1` | `private-inference-overlay.zip` | Full 1,569-train proposal, development/validation gates; no score established |
| `20260920-deep-pilot-4090-v1` | `private-inference-overlay.zip` | 256 training, development evaluation, baseline-only submit |
| `20260920-deep-pilot-submit-4090-v1` | `pilot-submit-overlay.zip` | Actual 256 private inference, validation skipped |
| `20260920-deep-pilot512-4090-v1` | `pilot512-overlay.zip` | Nested 512 training, comparison with 256, baseline-only submit |
| `20260921-deep-pilot512-submit-4090-v1` | `pilot512-submit-overlay.zip` | Actual 512 private inference, compatible raw-cache reuse |

Each archive has a `.zip.sha256` sidecar. The base package uses
`private-files.sha256`; add-ons use `pilot-submit-files.sha256`,
`pilot512-files.sha256`, or `pilot512-submit-files.sha256`. Check the base and
the relevant add-on manifests. Published local release folders contain
`published.json`; do not rebuild over them or change their contents.

Launchers are in `scripts/gpu/`; package builders are in `scripts/package/`.
Detailed commands are in the [256 pilot](../runbooks/RUNBOOK-deep-pilot-4090.md),
[256 submission](../runbooks/RUNBOOK-deep-pilot-submit-4090.md),
[512 pilot](../runbooks/RUNBOOK-deep-pilot512-4090.md), and
[512 submission](../runbooks/RUNBOOK-deep-pilot512-submit-4090.md) runbooks.
They reproduce the historical experiments, not a recommendation to redeploy them.

## Cache reuse and verification

`inputs.json` binds code, data, models, passage settings, precision and batch.
Each cache has `resume.json` and committed `part-*.json` shards. Deep raw shards
normally contain five questions. `fusion.json` binds weights to training IDs and
raw-score digest. `calibration.json` binds the selected policy to development
score digest. The direct-submission modules additionally pin the exact completed
pilot's input, fusion and development hashes.

- The 512 training importer validates all 256 existing raw training rows,
  including the final one-row shard, and all 429 development raw rows. Only the
  256 additional training questions require new pointwise model scoring.
- Development duo comparisons must be recomputed after changing fusion because
  the shortlist and prior can change. The learned weights themselves are CPU work.
- The 512 private continuation can import uploaded 256 private raw scores after
  checking identical scoring settings and replaying features/evidence. Fusion
  weights do not affect those raw scores. It computes missing raw shards and
  reruns duo comparisons; it does not reuse the 256 final rankings.
- A private import is a snapshot of completed uploaded shards, not live sharing
  throughout inference. The snapshot is refreshed on `run`; its import count is
  recorded in `private-reuse.json`.
- Existing valid destination shards are retained on resume. Identity mismatches
  stop the run; do not edit markers or hashes to bypass them.

Local tests verified partial-shard reuse, no duplicate model calls on resume,
private formatting/provenance, skipped stages and failure backup handling.
A real five-question private raw shard also passed feature/evidence replay for
512 reuse. These checks do not validate predictive quality; the user ran the
GPU jobs, and the private outcomes are the regressions recorded above.

## GPU environment and resource expectations

The 4090 runners require Linux x86_64, RTX 4090 24 GB, CUDA **12.8 devel**, and
Python 3.10–3.12 (prefer 3.12). Pinned Torch is `2.7.1+cu128`; FlashAttention is
`2.8.3`. Model checks require unquantized BF16. Use `all` for a fresh environment
and `run` to resume the installed workspace without reinstalling.

The fresh deep launchers check **80 GiB free space**. Measured restored models,
indexes, baseline cache and code total roughly 9.7 GiB. Python/CUDA packages,
build/download caches and new evidence caches add substantial disk use. The
earlier 30–50 GiB working estimate and the build peak were not benchmarked.
Allocate for actual free space after the base image, not just the rental's disk label.

Observed raw collection was about 19.6 s/question on the training pilot and
about 14.5 s/question early in private inference. These are different workloads
and running averages, not universal throughput. Model startup affects early
averages. A collection ETA excludes later comparisons, CPU checks and backups.
The original three-hour total estimate was too optimistic when measured scoring
was slower; use measured stage timings and state what remains unmeasured.

## Logs, failures and instance lifecycle

GPU workspace convention: `/workspace/DSC-2026-pilot-4090`.
Local logs: `outputs/private/<run-id>/run.log` and `transfer-errors.log`.
Cloud status: `runs/private/<run-id>/logs/STATUS.txt`; artifacts are in the sibling
`artifacts/`. Setup/restore can log under `assets-<run-id>` or a setup attempt.

The observed `www.googleapis.com: i/o timeout` was a DNS/network failure. One
pilot exited after committing all 256 training rows because its stage-boundary
backup failed. The checkpoints were still local and were reused on resume.
Inspect the transfer log and check connectivity before restarting. Do not delete
the instance or recompute completed work as a first response.

Restore-indexes downloads roughly 2.12 GB. A 2.0 GiB directory size does not prove
completion; check the restore log. The default transfer bound is 10 minutes,
artifact uploads use a 20-minute default, and the 512 private reuse download
has a 30-minute bound. Background upload failure can retry; a stage/final backup
failure can stop the launcher. Do not silently disable backup to hide failures.

The Vast helper is armed after setup/restore and asks for an API key when needed.
Confirm `AUTO-STOP ARMED` has the correct instance ID. It waits for successful
exit, validates both ZIPs, verifies cloud backup and then requests STOP. Failed
runs/uploads/API calls leave the instance running. STOP preserves disk and its
storage charges; it is not destruction. A `completed` line alone does not prove
the final backup or Vast STOP succeeded.

Separate GPUs may run different run IDs concurrently. Never run two writers to
the same cloud run. Workspace `flock` protects one local checkout, not two hosts.
Finish the prior run and its stop watcher before reusing that same instance.

In tmux: Ctrl+B, release, then C opens another shell; P returns to the previous
window. Ctrl+B then D detaches and may close a terminal that was attached directly
to tmux. Reattach with `tmux attach -t <session>`; list sessions with `tmux ls`.

## Repository and sharing boundaries

All `.sh` files are intentionally ignored by Git. Generated reports, `analysis/`,
models, caches, ZIPs and credentials are ignored too. The broad `analysis/`
pattern also ignores most `scripts/analysis/` helpers unless already tracked.
Do not assume the handoff's ignored files exist in a fresh clone. Obtain release
artifacts and share selected source/report files deliberately under the project's
existing repository policy. Do not force-add secrets or blanket-add ignored trees.

Only DSC cloud roots (`DSC-2026`, `DSC-2026-results`) are in scope. **Do not touch
AIC or NCKH.** Keep rclone configurations private. These notes do not publish
anything to GitHub, send messages to teammates or submit files to the competition.
