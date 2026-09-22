# Visible launcher and integrated auto-stop

This launcher replaces only startup orchestration. It uses the unchanged,
verified `20260921-fulltext-retrain-4090-v1` training ZIP and the same workspace,
run ID, frozen models and checkpoints. It never unzips over modified files.

The first launcher could return silently after switching to an existing tmux
session. An existing session can be an idle shell left after failure, so that
was not adequate evidence that GPU work had started. Version 2 checks actual
runner processes, follows the log in the current terminal, and prints failures.
It never switches tmux clients or nests an interactive tmux attachment.

Run in a root terminal on the same RTX 4090 rental:

```bash
set -euo pipefail
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | bash
fi
apt-get update
apt-get install -y nano tmux unzip build-essential python3-venv
umask 077
mkdir -p "$HOME/.config/rclone"
nano "$HOME/.config/rclone/rclone.conf"
chmod 600 "$HOME/.config/rclone/rclone.conf"
export RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
mkdir -p /workspace/dsc-retrain-launcher-v2
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-launcher-v2 /workspace/dsc-retrain-launcher-v2 --progress
cd /workspace/dsc-retrain-launcher-v2
sha256sum -c files.sha256
python3 launch_retrain.py
```

In nano, paste the complete config (or retain the working existing config), then
Ctrl+O, Enter, Ctrl+X. Enter the Vast API key only at the hidden terminal prompt.
The helper detects the instance ID, or asks for the instance ID if unavailable.
It verifies that identity with a read-only API request before arming.

The launcher checks pinned package versions and CUDA availability. A complete
existing environment uses `run`; an incomplete/missing environment uses `all`.
Fresh setup requires Linux x86_64, Python 3.10–3.12, CUDA **12.8 devel**, RTX 4090,
and **80 GiB free**. Resume requires **30 GiB free**. The actual values and errors
are printed in the current terminal before a new job starts. A rental with less
free disk must be expanded before running setup; this launcher does not delete
data to create space or silently lower the checks.

If the old run is active, it is preserved, its watcher is armed/reused, and the
launcher follows its latest log. If only an idle tmux shell remains, the launcher
starts a new job in `dsc-retrain-v2` after checking the workspace lock. Existing
idle sessions are not killed. Completed runs are checked and stopped instead of
retrained. A failed/incomplete attempt resumes its existing checkpoints.

A new attempt arms auto-stop before launching training. The watcher ignores
completion markers predating that attempt. A correctly matched live watcher can
be reused; an unverified watcher or workspace mismatch stops the launcher rather
than silently claiming protection. Setup/training/API/backup failure leaves the
instance running. A successful baseline fallback can stop just like a successful
candidate submission. STOP preserves disk; storage charges continue.

New startup logs are under
`outputs/launcher/duo-fulltext-retrain-4090-v1/<attempt>/launcher.log`.
`status.json` records the process and exit status. A best-effort final copy of
the launcher log goes to the DSC run's `logs/launcher/` directory. Existing
setup and training cloud logging is unchanged. The watcher verifies the actual
submission and all training artifacts before STOP; a copied log alone is not
sufficient.

Wait for `AUTO-STOP ARMED` or the verified `AUTO-STOP ALREADY ARMED`, followed by
`Job started ...` or `Existing job is running ...`. Live logs follow immediately.
Ctrl+C stops this log view only; the tmux job and watcher continue. Reopen the
same view with `python3 /workspace/dsc-retrain-launcher-v2/launch_retrain.py`.

This release is covered by CPU orchestration/failure tests and checksum checks.
The Linux tmux/GPU/API integration still needs observation on the actual rental;
no local test is represented as a live GPU run or a live STOP request.
