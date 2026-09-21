# Focused evidence and citation rescue on RTX 4090

Uses the existing unquantized BF16 duo checkpoint with FlashAttention 2. No
training, quantization or new model parameters. Separate workspace/run/release;
the scored 0.9510 submission is the fallback.

Requirements: RTX 4090 24 GB, Linux x86_64 CUDA **12.8 devel**, Python 3.10–3.12
(3.12 preferred), 32 GB RAM and at least 60 GiB free disk. Default batch: four.

## One Bash block

Run as root on the rental instance. Paste your rclone configuration in nano;
save Ctrl+O, Enter, then Ctrl+X. The remote must be named `gdrive`.

```bash
(
set -euo pipefail
apt-get update -qq
apt-get install -y -qq curl ca-certificates unzip nano tmux python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
mkdir -p /workspace/DSC-2026-rescue-4090
cd /workspace/DSC-2026-rescue-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-retrieval-rescue-4090-v1
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-retrieval-rescue-4090.sh all
else
  tmux new-session -s rescue4090 'AUTO_STOP=1 bash scripts/gpu/gpu-retrieval-rescue-4090.sh all'
fi
)
```

Already inside tmux is handled automatically. Detach with Ctrl+B, D. Reattach
to the fresh session with `tmux attach -t rescue4090`. If using an existing
session, reattach to its existing name instead. Do not interrupt inference.

If prompted, enter the Vast API key in the hidden prompt. Wait for
`AUTO-STOP ARMED`. The watcher is armed after setup/restore and before scoring;
setup/restore failures happen before arming. Compute stops only after successful
exit, complete valid ZIPs, and verified rclone backup. Failures leave the host
running. STOP preserves disk, so storage charges continue.

## What runs

`all`: setup -> restore -> prepare -> development -> calibrate -> validation ->
validate -> private -> submit. `run` resumes in an installed environment;
individual stages are also accepted, and `env` installs dependencies only.

- Reuses the exact 429 completed full-text development baseline rows. It does
  not repeat the failed full-text relevance scorer or refit its weight grid.
- Takes the current top eight and up to eight additional corpus-citation
  candidates. Focused passage selection uses bigrams, within-document rarity,
  diversity and the unchanged 2,048-token pair budget.
- Scores every contender against the original top five, including both A/B
  orders. At most 65 unordered pairs (130 model sequences) per question. It
  never fabricates probabilities between unmeasured challenger pairs.
- Development selects among baseline and four replacement policies: focused
  candidates alone or focused plus citations, each with 0.75/0.90 direct-win
  threshold. A challenger must also beat the victim's mean win rate by 0.05.
  At most one replacement per question, confined to original ranks four/five.
  The top three remain protected; this bounds both changes and recoverable misses.
- Ties keep baseline. If development retains baseline, skip validation/private
  GPU scoring and package the existing answers.
- Otherwise collect the missing **full-text baseline** for 1,050 validation
  questions, then the new challenges. This avoids judging a full-text private
  system against the old short-passage validation baseline.
- Only a strictly positive validation recall difference enables private scoring.
  Report paired wins/losses and bootstrap interval; the interval is reported,
  not an additional significance requirement. Private labels never fit policy.
- Submit writes both V9 and duo ZIPs. Inspect `submission-choice.json` to learn
  whether new answers were deployed. V9 remains the frozen original result.

No private score gain has been measured. Citation-pool coverage gains from the
CPU audit do not establish ranking accuracy. Historical validation has been
reused; a positive validation result still does not guarantee a leaderboard gain.

## Logs, outputs and time

Run: `private-retrieval-rescue-4090-v1`.

Local live log:
`outputs/private/private-retrieval-rescue-4090-v1/run.log`.

Cloud root:
`gdrive:DSC-2026-results/runs/private/private-retrieval-rescue-4090-v1/`.
Latest status: `logs/STATUS.txt` (about every 60 seconds).
ZIPs/caches/reports: `artifacts/`.
ZIP names: `submission-private-duo.zip`, `submission-private-v9.zip`.

Resume with the same workspace, batch and run ID:

```bash
cd /workspace/DSC-2026-rescue-4090
AUTO_STOP=1 bash scripts/gpu/gpu-retrieval-rescue-4090.sh run
```

Runtime is not GPU-benchmarked yet. Fewer measured pairs should reduce comparison
work, but focused CPU passage retrieval adds overhead. Budget about **1–2 hours
after setup** if development keeps baseline, and **8–12 hours after setup** if
all stages proceed, including the missing validation baseline. These are broad
planning estimates from the previous 10.5 seconds/question, not guarantees.
Use the measured per-stage ETA once the first shards finish.

Cloud restoration is additive and confined to DSC. Do not copy this overlay
over the scored or already-running prior workspace. Checkpoints reject changed
code, assets, labels, precision, candidate plans or batch sizes.
