# Duo full-text evidence v2

This is an inference experiment using the existing duo checkpoint. It adds no model parameters and does not train on private answers. The previous comparator weights, calibration strength, V9 assets, and attention backends remain frozen. A higher leaderboard score is not yet established; the old strength has not been recalibrated for the new evidence distribution.

Changes:

- A separate full-corpus SQLite FTS index covers every source word. This recovers pre-article/appendix text omitted by structural parsing without replacing V9's chunk IDs or dense vectors.
- Up to two additional candidates per question require a contiguous six-word match and at least 55% distinct query-token coverage in a source passage. They receive a low prior and must win comparator comparisons; no manual answer is injected.
- Document labels accompany query-centered passage spans. Actual tokenizer budgets are enforced before pair construction, so the original per-passage prefix quotas cannot discard the selected clause a second time.
- Each duo shard records recovered candidates, source window offsets, packed text and exact document token IDs. The template, question and separators remain in the frozen pair tokenizer.
- A compatible previous V9 cache can be imported, avoiding a second V9 GPU run. Question bytes, runtime, packages, assets and source signature must match. Old duo comparison shards cannot be reused.

## Package

Build locally:

```powershell
.venv\Scripts\python.exe scripts/package/package-private.py --fulltext
```

Output: `releases/private/20260919-duo-fulltext-v2/private-inference-overlay.zip`. It includes the frozen corpus JSONL gzip needed to build the side index, but excludes model weights and credentials. All frozen module hashes are checked by the packager.

## Existing compatible H100 environment

Extract the full-text package into the workspace and check its manifest:

```bash
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
bash scripts/gpu/gpu-duo-fulltext.sh all
```

Stages: restore assets and the previous private V9 cache → build full-text index on CPU → preflight → import V9 → rerun duo → submit/verify. Logs and outputs upload through the existing rclone worker to `gdrive:DSC-2026-results/runs/private/private-duo-fulltext-v2/`. Only DSC paths are used.

The default cache source is `private-v9-duo-9da4e0cb-v1`. Override `V9_SOURCE_RUN_ID` if the scored V9 run has another ID. Set it to an empty string to compute V9 again instead of restoring a cache:

```bash
V9_SOURCE_RUN_ID='' bash scripts/gpu/gpu-duo-fulltext.sh all
```

For a fresh machine, install the pinned environment first:

```bash
bash scripts/gpu/gpu-private-setup.sh env
bash scripts/gpu/gpu-duo-fulltext.sh all
```

The setup currently requires an H100/CUDA 12.8 development image. The saved duo uses FlashAttention2. Do not silently switch attention backends.

Use a new `PRIVATE_RUN_ID` for changed policy/code/inputs. Resume an unchanged run with the same command. An interrupted side-index build leaves `.building`; select a new `EVIDENCE_INDEX` path to restart that build. Completed side indexes are never overwritten. Allow at least several additional GiB for the raw corpus and full-text index.

Outputs:

```text
outputs/private/private-duo-fulltext-v2/
  inputs.json, audit.json, v9-import.json
  cache/v9/part-*.json
  cache/duo/part-*.json        # includes evidence traces
  submission-private-v9.zip  # original baseline
  submission-private-duo.zip # new experimental evidence policy
```

Both ZIPs contain only `submission.json`. Verify the new duo separately; retaining the baseline makes any regression reversible. Candidate expansion can increase a 16-document tournament from 240 ordered comparisons to 306, approximately 28% more comparisons, plus CPU retrieval. Actual runtime depends on candidate additions, batch padding, and cache availability.

## Verification limits

CPU regression tests cover source coverage, late-clause visibility, wrong-scope lexical distractors, tokenizer limits, cache imports and frozen-asset rejection. The sampled private audit is diagnostic, not a labeled validation set. Do not report its relevance judgments as measured recall, and do not tune thresholds against leaderboard feedback.
