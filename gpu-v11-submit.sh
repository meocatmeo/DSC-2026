#!/usr/bin/env bash
# Add submission to a completed older V11 run without recollecting training data.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec 9>"$ROOT/.v11-run.lock"
flock -n 9 || { echo 'V11 is already running in this workspace.'; exit 1; }
RESULTS="${BASE_RESULTS:-gdrive:DSC-2026-results}"
RUN_ID="${RUN_ID:-$(cat outputs/.v11-active-run)}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export OUT="${OUT:-outputs/v11-$RUN_ID}"
export PY="${PY:-$ROOT/.venv/bin/python}"
export BUNDLE_BATCH="${BUNDLE_BATCH:-1}"
LOG="$ROOT/outputs/v11-submit-$RUN_ID.log"
RUN_RESULTS="$RESULTS/v11/$RUN_ID"
source "$ROOT/gpu-v11-runtime.sh"
source "$ROOT/logship.sh"
finish_submit() {
  local code=$?
  trap - EXIT INT TERM
  stage_now "final-backup status=$code"
  stop_checkpoint_worker
  backup_checkpoints || code=1
  backup_v11 || code=1
  kill "$LOGSHIP_PID" 2>/dev/null || true
  wait "$LOGSHIP_PID" 2>/dev/null || true
  stage_now "exited status=$code"
  ship_logs || code=1
  exit "$code"
}
trap finish_submit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_checkpoint_worker
stage_now submission
if [ ! -f public-official.json ]; then
  rclone copyto "${REMOTE:-gdrive:DSC-2026}/public-official.json" public-official.json
fi
bash gpu-v11.sh submit
backup_v11
echo "Submission uploaded: $RESULTS/v11/$RUN_ID/artifacts/submission-v11.zip"
