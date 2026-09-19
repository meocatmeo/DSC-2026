# RTX 4090: faster full-text evidence preparation

This release replaces repeated n-gram set construction with an equivalent contiguous-token matcher. All phrase thresholds, candidate limits, passage budgets and model scoring remain unchanged. It includes no manual overrides or experimental boundary-reranking policy.

The matcher was approximately 8x faster on 500 corpus windows, with identical results. On six full development questions, evidence preparation was only about 6% faster in total because SQLite retrieval and tokenization still dominate. These are CPU measurements, not a 4090 end-to-end benchmark or a measured accuracy gain. Candidate lists, fallback priors, packed evidence text, token IDs and traces matched on all six questions. Randomized equivalence tests cover repeated words, Unicode normalization and the 24-token cap.

Follow [the 4090 setup guide](RUNBOOK-duo-4090.md) for system dependencies and rclone configuration. Use this separate release/run ID to avoid mixing resume fingerprints:

```bash
mkdir -p /workspace/DSC-2026-4090-fast
cd /workspace/DSC-2026-4090-fast
REMOTE=gdrive:DSC-2026-results/releases/private/20260919-duo-fulltext-v2-4090-fast1
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
tmux new -s duo4090fast
```

Inside tmux:

```bash
bash scripts/gpu/gpu-duo-4090-fast.sh all
```

Existing compatible environments can use `run` instead of `all`. If a previous inference run is active, let it finish; do not extract a changed package into its working directory. The old release remains available.

Default cloud results: `gdrive:DSC-2026-results/runs/private/private-duo-fulltext-v2-4090-fast1/artifacts/`.

Both submission ZIPs and logs are backed up automatically. No retraining or competition-site upload occurs. Use a new run ID for any batch/GPU/code change.

Build: `python scripts/package/package-private.py --fulltext --gpu 4090 --fast-evidence`.
