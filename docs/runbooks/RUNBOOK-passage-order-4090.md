# Passage-order development experiment on a fresh 4090

This release evaluates 429 development questions using both frozen comparators.
It does not retrain, run validation/private inference, submit answers, or replace
the best 0.9523 release. GPU execution remains to be tested on the rental.

Use an RTX 4090 with 24 GB VRAM, a CUDA 12.8 **devel** image, Python 3.12 and at
least 60 GiB free disk for the existing pinned environment installer. Two model
downloads total approximately 2.4 GB; no retrieval indexes or corpus are needed.
FlashAttention 2.8.3 is required, with no quantization or silent SDPA fallback.

## Start

In the rental's root terminal, install rclone and paste the complete existing
configuration using nano. Save with Ctrl+O, Enter; exit with Ctrl+X.

```bash
curl -fsSL https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone
nano ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
rclone copyto gdrive:DSC-2026-results/releases/passage-order/20260922-passage-order-4090-v1/start.sh /tmp/dsc-passage-start.sh --progress
bash /tmp/dsc-passage-start.sh
```

The release's start script pins the ZIP checksum. The user-facing command also
includes the generated start-script checksum, recorded in `release.json` alongside
the archive digest. Downloads and uploads are restricted to DSC paths. Config
files and model weights are never copied into run logs or output backups.

The bootstrap starts detached tmux session `dsc-passage-order`. It returns to the
terminal after starting the job. View it with:

```bash
tmux attach -t dsc-passage-order
```

If already inside tmux, use `tmux switch-client -t dsc-passage-order`. Existing
sessions are not replaced, and a workspace lock prevents overlapping runs.

## Stages, resume and results

The `all` stage installs the matching environment, restores both checkpoints and
prepared inputs, verifies their hashes, scores two questions per model, resumes
the complete 429-question score pass, then writes the paired evaluation report.
Every model load first tests the unchanged first question against its saved
predictions. A replay mismatch or any integrity failure stops the run.

Workspace: `/workspace/DSC-passage-order-4090-v2`.

Once setup has succeeded, resume after interruption with:

```bash
cd /workspace/DSC-passage-order-4090-v2
bash scripts/gpu/gpu-passage-order-4090.sh run
```

Resume checks the same model/plan identities and reuses completed per-question
scores. Interrupted question files remain uncommitted and are recomputed. Stage
options are `all`, `env`, `run`, `smoke`, `score`, `evaluate`.

Local log: `outputs/passage-order/passage-order-4090-v2/run.log`.
Local report: `outputs/passage-order/passage-order-4090-v2/evaluation.json`.
Local individual changes: the same directory's `changes.json`.

Cloud root:
`gdrive:DSC-2026-results/runs/passage-order/passage-order-4090-v2`.
Logs upload under `logs/<attempt>/`; reports and committed scores under `artifacts/`.
The existing runtime takes log snapshots and backs up every 60 seconds, at stage
boundaries and on exit. Temporary files are excluded. Failed background transfers
retain local files and retry; failed stage/final transfers produce a nonzero exit.

```bash
rclone copyto gdrive:DSC-2026-results/runs/passage-order/passage-order-4090-v2/artifacts/evaluation.json evaluation.json --progress
```

Estimated scoring time is 1.5–2.5 hours using prior 4090 throughput. Installation,
FlashAttention compilation and downloads add variable time; this release has no
new GPU benchmark yet. Review the first logged throughput before treating that
estimate as firm. No Vast account, instance ID or auto-stop watcher is configured
by this launcher; stop the rental after the completed report and backup are verified.

## Verification

The packager reloads all 429 prepared questions, checks code hashes, verifies ZIP
membership/bytes and writes checksums. Source Python bytes are preserved, including
line endings, so prepared code identities survive extraction. Bash files use LF.

Tests exercise real Bash stage dispatch with GPU and cloud commands stubbed,
including scoring failures and final backup. CPU tests cover evidence preservation
and comparison integrity. A separate extracted-release check loads all inputs and
runs the packaged tests. These checks do not substitute for the GPU smoke test.
