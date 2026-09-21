# Cheaper deep-pool pilot on RTX 4090

This separate run scores 256 fixed training questions (previously 1,569), fits
the 23 fusion weights, evaluates all 429 development questions, backs up and
stops. It does not automatically run validation or private inference, even if
development improves. BF16, FlashAttention 2, top-100 retrieval and four new
duo candidates are unchanged. Smaller training can weaken fusion; this is a
screening experiment, not evidence of a private score increase.

The training subset is fixed by salted normalized-question SHA256, independent
of labels, scores and development results. Training excludes all held-out IDs
and normalized question aliases. All development questions remain included.

Requirements: RTX 4090 24 GB, CUDA 12.8 devel, Python 3.10–3.12 (prefer 3.12),
32 GB RAM, 80 GiB free disk. Use a fresh workspace and run as root.

```bash
(
set -euo pipefail
apt-get update -qq
apt-get install -y -qq curl ca-certificates unzip nano tmux python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh | bash
mkdir -p ~/.config/rclone
nano ~/.config/rclone/rclone.conf
chmod 600 ~/.config/rclone/rclone.conf
mkdir -p /workspace/DSC-2026-pilot-4090
cd /workspace/DSC-2026-pilot-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-deep-pilot-4090-v1
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot-4090.sh all
else
  tmux new-session -s pilot4090 'AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot-4090.sh all'
fi
)
```

Paste the rclone config into nano, save with Ctrl+O, Enter, Ctrl+X. Enter the
Vast API key at the hidden prompt and wait for AUTO-STOP ARMED. Setup and
restore occur before arming. After successful completion the existing watcher
verifies backup before STOP; run/upload/API failure leaves the instance running.
STOP preserves the disk, so storage charges continue.

Stages: setup → restore → prepare → training → fit → development → calibrate
→ submit (unchanged baseline ZIPs only) → backup → STOP.

The submit stage preserves the existing V9 and 0.9510 duo answers for the
watcher's submission integrity check. These are **not new improved submissions**.
Review `calibration.json` and `pilot-report.json`. No validation result is claimed.
There is no automatic continuation to expensive inference.

Local log: `outputs/private/private-deep-pilot-4090-v1/run.log`.
Cloud root: `gdrive:DSC-2026-results/runs/private/private-deep-pilot-4090-v1/`.
Live status is `logs/STATUS.txt`; reports, caches and baseline ZIPs are in
`artifacts/`. Only DSC remote paths are used.

Resume an installed workspace using
`AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot-4090.sh run`.
The original full-release run ID/assets must not be reused for this subset.

No GPU benchmark is available. At illustrative speeds of 10 seconds per raw
collection question and 8 seconds per new-comparison question, compute takes
about **2.9 hours**, versus 6.5 hours through development previously. Setup,
restore, CPU fitting and uploads are additional. Use measured shard times to
update the estimate; this is not a three-hour time limit or completion guarantee.
