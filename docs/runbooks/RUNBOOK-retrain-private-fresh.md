# Restore private submission on a replacement GPU

The old machine is unavailable. The DSC cloud contains the trained `control`
checkpoint (about 1.21 GB) and frozen private evidence cache (about 596 MB).
The saved reference manifest identifies 208 private shards. A fresh host checks
all expected shard checksums, rows and model files before inference. Extra cloud
cache files are not adopted. No retraining, validation, new policy selection,
retrieval index rebuilding, or download of the original retriever models occurs.

Use an RTX 4090, Linux x86_64, Python 3.10–3.12 (3.12 recommended), CUDA **12.8
devel** including `nvcc`, and at least **80 GiB free disk**. Provisioning 100 GB
or more allows room for the image and pinned environment installer. Keep the old
instance's disk until the new output is safely backed up if it remains available.

In the new root terminal:

```bash
(
set -euo pipefail
apt-get update
apt-get install -y curl nano tmux unzip build-essential python3-venv
curl -fsSL https://rclone.org/install.sh | bash
umask 077
mkdir -p "$HOME/.config/rclone"
nano "$HOME/.config/rclone/rclone.conf"
chmod 600 "$HOME/.config/rclone/rclone.conf"
export RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
mkdir -p /workspace/dsc-retrain-private-fresh
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-private-submit-v2 /workspace/dsc-retrain-private-fresh --progress
cd /workspace/dsc-retrain-private-fresh
sha256sum -c files.sha256
python3 launch_retrain.py --fresh-private
)
```

Paste the complete existing rclone config into nano, then Ctrl+O, Enter, Ctrl+X.
Enter the Vast API key at the hidden prompt. The helper identifies and verifies
the **new instance**, then arms STOP before the tmux job starts. The watcher waits
for this private submission run, so environment/restore stage completion cannot
stop it early. On successful inference, valid ZIP and verified cloud backup it
requests STOP. Failure leaves the instance running for recovery.

Sequence: frozen code download/check -> arm STOP -> install pinned BF16/Flash
Attention environment -> restore trained control and private evidence -> verify
every required shard -> private inference -> pack ZIP -> backup -> STOP.

Completed private shards, if any, are restored from the candidate run's cloud
folder; old hidden exit markers are excluded. Training source files and baseline
submission stay unchanged. The separate candidate run records that deployment
was explicitly requested without a passing validation gate.

Output:
`gdrive:DSC-2026-results/runs/retrain/duo-fulltext-retrain-private-control-4090-v1/artifacts/submission-private-duo.zip`

The corresponding cloud `logs/` contains restore/inference progress; environment
logs also upload under `runs/setup/private/`. The launcher follows all output in
the current terminal. Rerun the same launcher command to follow/resume. Ctrl+C
ends only the log view. Never rerun the old training `all` command for this task.

Cloud presence and sizes were checked locally. CPU tests cover fresh restore,
private-only stage order, backup on failures, source preservation, decision
metadata, and auto-stop checks. Actual GPU kernels and API STOP are verified only
on the rented host; no local mock result is claimed as a live GPU run.
