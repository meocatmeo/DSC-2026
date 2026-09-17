#!/usr/bin/env bash
# Autonomous V11 restore -> benchmark -> collect -> fit -> compare -> cloud backup.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# A second launch must not overwrite checkpoints or share GPU memory with this run.
if [ "${V11_LOCK_HELD:-}" != "$ROOT" ]; then
  exec 9>"$ROOT/.v11-run.lock"
  flock -n 9 || { echo 'V11 is already running in this workspace.'; exit 1; }
  export V11_LOCK_HELD="$ROOT"
fi
REMOTE="${REMOTE:-gdrive:DSC-2026}"
BASE_RESULTS="${BASE_RESULTS:-gdrive:DSC-2026-results}"
mkdir -p outputs
if [ -z "${RUN_ID:-}" ] && [ -f outputs/.v11-active-run ]; then
  RUN_ID="$(cat outputs/.v11-active-run)"
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID'; exit 2; }
printf '%s\n' "$RUN_ID" > outputs/.v11-active-run
export OUT="${OUT:-outputs/v11-$RUN_ID}"
export PY="${PY:-$ROOT/.venv/bin/python}"
export LIMIT="${LIMIT:-2000}"
export BUNDLE_BATCH="${BUNDLE_BATCH:-2}"
RESULTS="$BASE_RESULTS"
RUN_RESULTS="$BASE_RESULTS/v11/$RUN_ID"
LOG="$ROOT/outputs/v11-$RUN_ID.log"
command -v rclone >/dev/null || { echo 'rclone with gdrive remote is required'; exit 1; }
mkdir -p "$OUT"
source "$ROOT/gpu-v11-runtime.sh"
source "$ROOT/logship.sh"

finish_v11() {
  local code=$?
  trap - EXIT INT TERM
  stage_now "final-backup status=$code"
  stop_checkpoint_worker
  echo 'Final backup: saving unfinished question checkpoints before exit'
  backup_checkpoints || { echo 'FINAL CHECKPOINT UPLOAD FAILED'; code=1; }
  backup_v11 || { echo 'FINAL ARTIFACT UPLOAD FAILED'; code=1; }
  kill "$LOGSHIP_PID" 2>/dev/null || true
  wait "$LOGSHIP_PID" 2>/dev/null || true
  stage_now "exited status=$code"
  ship_logs || { echo 'FINAL LOG UPLOAD FAILED'; code=1; }
  exit "$code"
}
trap finish_v11 EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ship_logs  # Fail before expensive work if the upload channel is unavailable.
# Logs use the shared results/logs directory, like the other phase scripts.
# Keep checkpoints scoped by run, creating their parent before discovery.
rclone mkdir "$RUN_RESULTS"
RUN_DIRS="$(rclone lsf "$RUN_RESULTS" --dirs-only)"
if grep -Fxq 'artifacts/' <<< "$RUN_DIRS"; then
  echo 'Restoring saved V11 checkpoints'
  rclone copy "$RUN_RESULTS/artifacts" "$OUT" --ignore-existing --exclude '*.partial' --exclude '*.tmp' --exclude '**.rows/**'
  SAVED_DIRS="$(rclone lsf "$RUN_RESULTS/artifacts" --dirs-only --recursive)"
  for split in train validation public benchmark/train; do
    if { [ ! -f "$OUT/$split.jsonl.gz" ] || [ ! -f "$OUT/$split.manifest.json" ]; } && \
        grep -Fxq "$split.rows/" <<< "$SAVED_DIRS"; then
      rclone copy "$RUN_RESULTS/artifacts/$split.rows" "$OUT/$split.rows" --ignore-existing --exclude '*.partial' --exclude '*.tmp'
    fi
  done
else
  echo 'New run; no cloud checkpoints to restore.'
fi
start_checkpoint_worker

stage_now restore-inputs
echo "V11 run=$RUN_ID remote=$REMOTE results=$RUN_RESULTS output=$OUT"
echo 'Restoring required inputs from Drive'
# Restore only the active experiment's inputs, not all historical models.
if [ ! -f artifacts/splits/train.json ] || [ ! -f artifacts/splits/validation.json ]; then
  rclone copy "$REMOTE/artifacts/splits" artifacts/splits --include '*.json' || echo 'Split download failed; checking local inputs.'
fi
for split in train validation; do
  [ -s "artifacts/splits/$split.json" ] || { echo "Required split missing: $split"; exit 1; }
done
if [ ! -f corpus.jsonl.gz ]; then
  echo 'Downloading corpus.jsonl.gz'
  rclone copyto "$REMOTE/corpus.jsonl.gz" corpus.jsonl.gz --stats 30s --stats-one-line
fi
if [ ! -f public-official.json ]; then
  rclone copyto "$REMOTE/public-official.json" public-official.json
fi
mkdir -p analysis/v11/baseline
for f in v9-ensemble.run.json v9-stacker-ens.json; do
  if [ ! -f "analysis/v11/baseline/$f" ]; then
    rclone copyto "$BASE_RESULTS/outputs/$f" "analysis/v11/baseline/$f"
  fi
done
stage_now environment
echo 'Preparing Python environment'
bash gpu-setup.sh env
stage_now restore-models
for model in embedding reranker-g32; do
  if ! "$PY" -m legalir2.v11_preflight model "models2/$model" >/dev/null; then
    stage_now "download-$model"
    echo "Downloading $model from $BASE_RESULTS/models/$model (local model missing or incomplete)"
    rclone copy "$BASE_RESULTS/models/$model" "models2/$model" --transfers 4 || echo "Restore failed for $model; checking local files."
  fi
  if ! "$PY" -m legalir2.v11_preflight model "models2/$model"; then
    if [ "$model" = embedding ]; then
      ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; "$PY" -m legalir2 download --which embedding )
    else
      echo 'Trained reranker-g32 is unavailable locally and on remote; cannot substitute base weights.'
      exit 1
    fi
  fi
done
if ! "$PY" -m legalir2.v11_preflight model models2/partner_qwen >/dev/null; then
  stage_now download-partner_qwen
  echo 'Downloading partner_qwen (local model missing or incomplete)'
  # The base Qwen partner was not present in the historical model backup.
  ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    "$PY" -u -m legalir2 download --which partner_qwen )
fi
stage_now indexes
# Reuse current v9 indexes if present; fresh boxes build an internally consistent set.
if [ ! -f artifacts2/store.sqlite ]; then
  "$PY" -u -m legalir2 build-store --corpus corpus.jsonl.gz
fi
if [ ! -f artifacts2/lexical/bigram.npz ] || [ ! -f artifacts2/lexical/unigram.npz ]; then
  "$PY" -u -m legalir2 index-lexical
fi
if [ ! -f artifacts2/dense/manifest.json ]; then
  rclone copy "$BASE_RESULTS/dense" artifacts2/dense --transfers 4 || echo 'Dense restore failed; will rebuild if required.'
fi
# No historical corpus gate. Only the actual store/vector mapping must agree.
if ! "$PY" -m legalir2.v11_preflight dense; then
  stage_now rebuild-dense
  "$PY" -u -m legalir2 index-dense --embedding-model models2/embedding
fi
stage_now tests
"$PY" -m unittest tests2.test_evidence tests2.test_v11 tests2.test_v11_resume -v
stage_now benchmark
bash gpu-v11.sh benchmark
backup_v11
stage_now collect
bash gpu-v11.sh collect
backup_v11
# Completed splits skip row restore/upload. Do not race the background uploader by
# purging its remote directory while it can still be writing there.
stage_now fit
bash gpu-v11.sh fit
backup_v11
stage_now evaluate
bash gpu-v11.sh evaluate
backup_v11
stage_now submission
bash gpu-v11.sh submit
backup_v11
stage_now completed
echo "Completed. Results: $RUN_RESULTS/artifacts/comparison.json"
echo "Submission: $RUN_RESULTS/artifacts/submission-v11.zip"
echo "Monitor: rclone cat $RESULTS/logs/STATUS.txt"
