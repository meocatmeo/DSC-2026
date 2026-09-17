#!/usr/bin/env bash
# Download the staged V11 release; then hand over to its autopilot.
set -euo pipefail
ROOT="${ROOT:-/workspace/DSC-2026}"
REMOTE="${REMOTE:-gdrive:DSC-2026}"
command -v rclone >/dev/null || { echo 'Configure rclone gdrive first'; exit 1; }
mkdir -p "$ROOT"
exec 9>"$ROOT/.v11-run.lock"
flock -n 9 || { echo 'V11 is already running; release extraction was not attempted.'; exit 1; }
export V11_LOCK_HELD="$ROOT"
echo 'Downloading V11 release (progress every 30 seconds)'
timeout --signal=TERM --kill-after=10s 10m rclone copyto "$REMOTE/v11-release.zip" "$ROOT/v11-release.zip" \
  --contimeout 20s --timeout 2m --retries 2 --low-level-retries 3 --stats 30s --stats-one-line
python3 - "$ROOT" <<'PY'
from pathlib import Path
import sys, zipfile
root = Path(sys.argv[1]).resolve()
with zipfile.ZipFile(root / 'v11-release.zip') as z:
    for name in z.namelist():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Invalid archive path')
    z.extractall(root)
PY
exec bash "$ROOT/gpu-v11-auto.sh"
