# Full-text relevance and tournament consensus on RTX 4090

Separate experiment; the scored 0.9510 release is preserved. Uses the existing
Qwen3-Reranker-0.6B in its original yes/no relevance mode on every candidate's
full-text passages. No new model parameters or retraining. Duo's saved private
comparisons are reused. Pointwise and comparator models load sequentially.
Includes the Borda, PageRank and head-to-head ideas from `thong/e091868`,
plus a corrected Copeland win count. No quantization code was present in that
branch; this release retains the existing BF16 checkpoints.

Requirements: RTX 4090 24 GB, Ubuntu/Debian CUDA **12.8 devel** image, Python
3.10-3.12 (3.12 preferred), at least 60 GiB free disk and 32 GB system RAM.
FlashAttention 2.8.3 and recorded package versions install through the existing
setup script. Default GPU batch is 4; no silent attention fallback.

## Fresh instance: one Bash block

Paste your rclone configuration in nano; save Ctrl+O, Enter, exit Ctrl+X.
The Vast key, if needed, is entered locally in a hidden prompt.

```bash
(
set -euo pipefail
apt-get update -qq
apt-get install -y -qq curl ca-certificates unzip nano tmux python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
mkdir -p /workspace/DSC-2026-prior-4090
cd /workspace/DSC-2026-prior-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-fulltext-prior-4090-v1
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
tmux new-session -s prior4090 'AUTO_STOP=1 bash scripts/gpu/gpu-fulltext-prior-4090.sh all'
)
```

Detach with Ctrl+B then D. Reattach: `tmux attach -t prior4090`.
Auto-stop is armed after source restoration and before GPU scoring. It stops
only after successful completion, both valid ZIPs, and verified cloud backup.
Failures leave the instance running. STOP preserves storage; destroy the
instance after verifying downloads to end storage charges. `AUTO_STOP=0` disables
arming. Setup failures occur before the main watcher is armed.

## Stages and deployment rule

`all`: environment -> restore models/indexes/scored source -> prepare ->
development -> calibrate -> validation -> validate -> private -> submit.

- Development: 429 labeled questions, new full-text comparator and relevance
  scores. Select among baseline, 12 prior/blend alternatives and six fixed
  consensus alternatives (two voting profiles at three full-text weights).
  If baseline wins, skip validation and private GPU scoring and package the
  original answers immediately.
- Validation: 1,050 labeled questions scored with the same evidence. Report
  recall, paired wins/losses and bootstrap interval. No weights fitted here.
- Private: only if validation recall strictly improves; score all candidates
  using the existing saved comparisons. No private labels used.
- Submit: produces both submission ZIPs. If validation does not improve,
  preserves the previous duo answers instead of running private GPU scoring.
  Check `submission-choice.json`: a ZIP is not evidence of an improvement.

The prior combines standardized full-text relevance logits and the old V9
prior. Tournament fitting uses that new prior; development chooses its weight
and final blend. The baseline option reproduces the current policy exactly.
Consensus policies require direct pairwise support before replacing an answer.
They consider every compared candidate, including full-text additions. No
document is rejected merely because it is absent from training answers.
All development/validation IDs and normalized duplicate questions are excluded
from the frozen training memory. The original `scripts/rerank_*.py` prototypes
are research references and are not used or shipped by this runner.
Validation is historically reused, and even a positive validation delta does
not guarantee a private score increase.

Use `run` to resume all stages in an installed environment. Individual stages:
`prepare`, `development`, `calibrate`, `validation`, `validate`, `private`,
`submit`. `env` installs dependencies only. Resume uses atomic shards and
code/input/model/evidence fingerprints. Use a new run ID for changed code or
batch size; never overwrite the 0.9510 workspace.

## Outputs and time

Run ID: `private-fulltext-prior-4090-v1`.
Local output: `outputs/private/private-fulltext-prior-4090-v1/`.
Cloud: `gdrive:DSC-2026-results/runs/private/private-fulltext-prior-4090-v1/`.
Logs: `logs/STATUS.txt`; outputs and resume caches: `artifacts/`.
ZIP names: `submission-private-duo.zip`, `submission-private-v9.zip`.

The 1,479 labeled full-text comparison questions are the expensive part. At the
previous run's approximately 12 seconds/question, that portion alone would be
about 4.9 hours. Setup, downloads, relevance scoring and upload add time. Budget
roughly 6-8 hours initially; this is an extrapolation, not a GPU benchmark of
the new run. Runtime logs update measured per-stage ETA. Private comparisons
are reused, saving their previous approximately seven-hour cost.
