# Top-100 retrieval and trained fusion on RTX 4090

Separate experiment: `private-deep-pool-4090-v1`. Existing model weights remain
BF16, with FlashAttention 2 and no quantization. The scored 0.9510 answers are
the fallback. No private improvement has been measured for this experiment.

Requirements: Linux x86_64, RTX 4090 24 GB, CUDA **12.8 devel** toolkit,
Python 3.10–3.12 (prefer 3.12), 32 GB RAM and **at least 80 GiB free disk**.
The expanded raw-score/evidence caches are larger than the previous run.

## One Bash block

Run as root. In nano paste the rclone config containing `[gdrive]`, then save
with Ctrl+O, Enter, Ctrl+X. Use the new workspace; do not overlay a running or
scored experiment.

```bash
(
set -euo pipefail
apt-get update -qq
apt-get install -y -qq curl ca-certificates unzip nano tmux python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
mkdir -p /workspace/DSC-2026-deep-4090
cd /workspace/DSC-2026-deep-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-deep-pool-4090-v1
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-deep-pool-4090.sh all
else
  tmux new-session -s deep4090 'AUTO_STOP=1 bash scripts/gpu/gpu-deep-pool-4090.sh all'
fi
)
```

If requested, enter the Vast API key at the hidden prompt. Wait for
`AUTO-STOP ARMED`. Setup/restore happen before arming. After successful submit,
the watcher verifies the cloud backup before requesting STOP. Any run, upload
or API failure leaves the instance running. Stopping preserves disk and storage
charges. Detach tmux with Ctrl+B, D; reattach with `tmux attach -t deep4090` if
this block created that session.

## What this run changes

1. Restore the frozen checkpoints, indexes and scored private baseline. Verify
   code/model/data hashes and replay all 2,080 private baseline answers.
2. Collect the real fused top 100 plus every preserved original duo candidate.
   No gold labels choose candidates. Retain raw scores, retrieval rankings,
   selected passage text, token IDs and normalized features.
3. Both existing rerankers score the expanded pool. G32 scores the selected
   fulltext-v2 passages separately at its saved context length; Qwen scores an
   exactly budgeted 2,048-token document bundle. Original duo passages are
   reused byte-for-byte when available. The failed focused-evidence selector
   and citation expansion are not used.
4. Fit 23 linear fusion coefficients on the frozen training partition only,
   with fixed 300 epochs and L2=0.001. Features include both rerankers, retrieval
   channels and the old V9 score/rank/presence. Normalization is identical for
   training and inference at the expanded depth. These coefficients are
   included conservatively in the total parameter audit.
5. On development, keep every original duo candidate and append at most four
   candidates admitted by learned fusion. Reuse every old comparison and run
   both A/B orders only for edges involving new documents. Fit the expanded
   tournament with the learned prior, retaining duo strength 0.25.
6. Select baseline or the new policy using development recall. Ties keep the
   baseline. If baseline wins, skip validation/private scoring and submit the
   existing answers. No confidence-threshold sweep is performed.
7. If development improves, collect the missing full-text validation baseline,
   then validate the same fixed new policy on all 1,050 validation questions.
   Only a strictly positive recall difference enables private scoring. The
   paired confidence interval is reported; it is not an additional gate.
8. Submit produces both ZIPs. V9 stays the frozen original submission. Duo uses
   the candidate only if it cleared validation; otherwise its answers exactly
   match the scored 0.9510 reference. Read `submission-choice.json`.

Development/validation identities and normalized question aliases cannot enter
fusion training. Existing neural rerankers may have seen calibration-training
questions, and validation has been historically reused; these limitations are
recorded. No model-weight fine-tuning occurs here. A larger embedding-training
intervention requires a separately validated experiment.

## Stages, logs and resume

`all`: setup -> restore -> prepare -> training -> fit -> development ->
calibrate -> validation -> validate -> private -> submit.
`run` resumes an installed environment; each named stage may also be invoked.

Local log: `outputs/private/private-deep-pool-4090-v1/run.log`.

Cloud root:
`gdrive:DSC-2026-results/runs/private/private-deep-pool-4090-v1/`.
Live status: `logs/STATUS.txt` (about every 60 seconds).
Submissions, caches and audits: `artifacts/`.

```bash
cd /workspace/DSC-2026-deep-4090
AUTO_STOP=1 bash scripts/gpu/gpu-deep-pool-4090.sh run
```

Reuse the same run ID, batch, assets and code to resume. Corrupt or mismatched
shards stop the run rather than silently consuming more GPU time. Training
scores are reusable; fusion fitting is CPU work. Default inference batch is 4.

## Cost expectations

This is a larger experiment than the one-hour focused-rescue test. It collects
training scores for roughly 1,500 questions before the 429-question development
gate. A full successful path also scores 1,050 validation and 2,080 private
questions, plus the missing validation baseline. Do not apply the previous
7.8-seconds/question figure to this new scoring stage.

No full GPU benchmark exists yet. Each collection/comparison stage prints its
own measured seconds per question and ETA. As an **illustration, not a forecast**,
at 10 seconds per collection question and 8 seconds per new-comparison question,
training plus development would take about 6.5 hours; all stages would take
about 25 hours plus setup, fitting and uploads. A GPU with weak CPU/disk/network
can take longer. Use the first training shards to reassess cost.

The launcher adds a disk-space precheck and retains automatic backup and STOP.
Only DSC remote paths are used; AIC and NCKH are outside this workflow.
