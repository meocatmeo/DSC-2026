#!/usr/bin/env bash
# Shared V11 transfer limits and checkpoint lifecycle. Source after ROOT/OUT are set.
rclone() {
  # A dead Drive connection must not keep a rented GPU idle indefinitely. `timeout`
  # also bounds listing/authentication, which transfer-only time limits do not.
  timeout --signal=TERM --kill-after=10s "${V11_TRANSFER_TIMEOUT:-30m}" \
    rclone --contimeout 20s --timeout 2m --retries 2 --low-level-retries 3 \
    --stats 30s --stats-one-line "$@"
}

backup_v11() {
  echo "Uploading completed V11 artifacts to $RUN_RESULTS/artifacts"
  rclone copy "$OUT" "$RUN_RESULTS/artifacts" --transfers 4 \
    --exclude '*.partial' --exclude '*.tmp' --exclude '**.rows/**'
}

backup_checkpoints() (
  local -a filters=(--filter '- *.partial' --filter '- *.tmp' --filter '- **.rows/index.jsonl' --filter '+ **/')
  local split snapshot active=0
  mkdir -p "$ROOT/outputs"
  snapshot="$(mktemp -d "$ROOT/outputs/.v11-checkpoint.XXXXXX")" || return
  # Only this invocation's flat temporary files are removed, never checkpoint rows.
  trap 'rm -f -- "$snapshot"/*.jsonl; rmdir -- "$snapshot"' EXIT
  "${V11_SNAPSHOT_PYTHON:-python3}" -m legalir2.v11_backup "$OUT" "$snapshot" || return
  # Snapshot before transferring rows. Any later appends wait for the next backup.
  for split in train validation public benchmark/train; do
    if [ -f "$snapshot/${split//\//-}.jsonl" ]; then
      active=1
      filters+=(--filter "+ /$split.rows/**")
      # Guard first, immutable rows next, frozen index last. A failed row transfer
      # must never publish an index claiming all referenced rows were uploaded.
      rclone copyto "$OUT/$split.rows/resume.json" "$RUN_RESULTS/artifacts/$split.rows/resume.json" || return
    fi
  done
  [ "$active" -eq 1 ] || return 0
  rclone copy "$OUT" "$RUN_RESULTS/artifacts" "${filters[@]}" \
    --filter '- **' --transfers 4 || return
  for split in train validation public benchmark/train; do
    if [ -f "$snapshot/${split//\//-}.jsonl" ]; then
      rclone copyto "$snapshot/${split//\//-}.jsonl" "$RUN_RESULTS/artifacts/$split.rows/index.jsonl" || return
    fi
  done
)

start_checkpoint_worker() {
  (
    delay=''
    # wait on the sleep asynchronously so shutdown needn't wait the whole interval.
    # A transfer already in progress finishes (bounded by timeout) before exit;
    # the foreground final backup then runs without a competing uploader.
    trap 'if [ -n "$delay" ]; then kill "$delay" 2>/dev/null || true; fi; exit 0' TERM INT
    while true; do
      sleep "${CHECKPOINT_EVERY:-300}" & delay=$!
      wait "$delay" || exit 0
      delay=''
      V11_TRANSFER_TIMEOUT=5m backup_checkpoints || echo 'Checkpoint upload failed; local rows retained.'
    done
  ) &
  CHECKPOINT_PID=$!
}

stop_checkpoint_worker() {
  if [ -n "${CHECKPOINT_PID:-}" ]; then
    kill "$CHECKPOINT_PID" 2>/dev/null || true
    wait "$CHECKPOINT_PID" 2>/dev/null || true
    CHECKPOINT_PID=''
  fi
}
