#!/usr/bin/env bash
# Self-logging: source this at the top of any phase script and it becomes watchable
# from anywhere, forever, without a second terminal.
#
#   ROOT=/workspace/DSC-2026
#   source "$ROOT/logship.sh"          # after ROOT is set, before any real work
#   stage_now "crossfit"               # call this when a stage begins
#
# What sourcing it does:
#   * captures EVERY line - the script, its children, python, tracebacks - into a log
#   * writes a STATUS.txt heartbeat with the current stage, elapsed time and GPU load
#   * ships both to Drive every 60 s, so `rclone cat .../logs/STATUS.txt` always works
#   * proves the upload channel works in the first two seconds rather than at hour three
#   * reports its own upload failures instead of swallowing them
#   * ships a final copy on exit, including after Ctrl-C or a crash
#
# This lived as copy-pasted blocks in three separate autopilots. A behaviour that every
# long run needs should be one file that every long run sources - the copies had already
# started to drift, and the oldest of them was the one silently failing to upload.
LOG_EVERY="${LOG_EVERY:-60}"
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
ROOT="${ROOT:-$(pwd)}"
mkdir -p "$ROOT/outputs"
LOG="${LOG:-$ROOT/outputs/run.log}"
LOGSHIP_STATE="$ROOT/outputs/.logship-state"
LOGSHIP_ERRORS="$ROOT/outputs/rclone-errors.log"
LOGSHIP_STARTED=$(date +%s)
LOGSHIP_FAILURES=0
# BASH_SOURCE inside a function resolves to this file, not the script that sourced it,
# so the name is captured here at source time while $0 still means the caller.
LOGSHIP_NAME="$(basename "${0:-run}")"

# Do this once, at the top. Per-command `| tee` only ever captures the commands someone
# remembered to pipe, which is never the traceback you actually needed.
exec > >(tee -a "$LOG") 2>&1

# The shipper runs in a subshell forked now, so it cannot see variables set later.
# State goes through a file or the heartbeat reports the run's opening moments forever.
stage_now() {
  printf 'stage=%s\nstarted=%s\nhost=%s\n' \
      "$1" "$LOGSHIP_STARTED" "$(hostname)" > "$LOGSHIP_STATE.tmp"
  mv "$LOGSHIP_STATE.tmp" "$LOGSHIP_STATE"
}
stage_now "starting"

ship_logs() {
  command -v rclone >/dev/null || return 0
  local stage started elapsed
  stage="$(sed -n 's/^stage=//p' "$LOGSHIP_STATE" 2>/dev/null || echo '?')"
  started="$(sed -n 's/^started=//p' "$LOGSHIP_STATE" 2>/dev/null || echo "$LOGSHIP_STARTED")"
  elapsed=$(( ($(date +%s) - started) / 60 ))
  {
    printf 'DSC-2026  %s\n' "$LOGSHIP_NAME"
    printf 'host            : %s\n' "$(hostname)"
    printf 'log             : %s\n' "$(basename "$LOG")"
    printf 'heartbeat (UTC) : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'running for     : %s h %s min\n' "$((elapsed / 60))" "$((elapsed % 60))"
    printf 'current stage   : %s\n' "$stage"
    if command -v nvidia-smi >/dev/null; then
      printf 'gpu             : %s\n' \
        "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
             --format=csv,noheader 2>/dev/null | head -1)"
    fi
    printf 'disk free       : %s\n' "$(df -h "$ROOT" | awk 'NR==2{print $4}')"
    printf '\n--- last 40 lines ---\n'
    tail -n 40 "$LOG" 2>/dev/null
  } > "$ROOT/outputs/STATUS.txt" 2>/dev/null

  # --ignore-checksum: the log is appended to WHILE it uploads, so its md5 legitimately
  # differs between read and verify. rclone calls that "corrupted on transfer", retries
  # and succeeds - correct, but it buries real errors under hours of false alarms.
  if ! V11_TRANSFER_TIMEOUT=2m rclone copy "$ROOT/outputs" "$RESULTS/logs" --ignore-checksum --max-depth 1 \
          --contimeout 20s --timeout 30s --retries 1 --low-level-retries 2 \
          --include "*.log" --include "STATUS.txt" --include "*.stats.json" \
          >>"$LOGSHIP_ERRORS" 2>&1; then
    LOGSHIP_FAILURES=$((LOGSHIP_FAILURES + 1))
    printf '[%s] log upload FAILED (%s in a row) - see outputs/rclone-errors.log\n' \
        "$(date +%H:%M:%S)" "$LOGSHIP_FAILURES" >> "$LOG"
    if [ "$LOGSHIP_FAILURES" = 1 ] || [ $((LOGSHIP_FAILURES % 20)) = 0 ]; then
      printf '  last rclone error:\n' >> "$LOG"
      tail -n 4 "$LOGSHIP_ERRORS" | sed 's/^/    /' >> "$LOG"
    fi
    return 1
  fi
  [ "$LOGSHIP_FAILURES" -gt 0 ] && printf '[%s] log upload recovered\n' \
      "$(date +%H:%M:%S)" >> "$LOG"
  LOGSHIP_FAILURES=0
  return 0
}

# Prove the channel before anything expensive. Discovering at hour three that nothing
# was ever uploaded is the failure this whole file exists to prevent.
if command -v rclone >/dev/null; then
  printf '[%s] testing the log channel to %s/logs ...\n' "$(date +%H:%M:%S)" "$RESULTS"
  if ship_logs; then
    printf '[%s] log channel OK - watch it with:\n' "$(date +%H:%M:%S)"
    printf '           rclone cat %s/logs/STATUS.txt\n' "$RESULTS"
  else
    printf '\n  WARNING: cannot upload logs to %s/logs.\n' "$RESULTS"
    printf '  The run continues and everything is still written to %s,\n' "$LOG"
    printf '  but you will not be able to watch it from off this box.\n\n'
  fi
else
  printf '\n  WARNING: rclone is not installed - logs stay local to this box.\n\n'
fi

(
  delay=''
  trap 'if [ -n "$delay" ]; then kill "$delay" 2>/dev/null || true; fi; exit 0' TERM INT
  while true; do
    sleep "$LOG_EVERY" & delay=$!
    wait "$delay" || exit 0
    delay=''
    ship_logs || true  # a transient Drive failure must not kill the heartbeat
  done
) &
LOGSHIP_PID=$!

logship_cleanup() {
  local code=$?
  trap - EXIT INT TERM
  kill "$LOGSHIP_PID" 2>/dev/null || true
  wait "$LOGSHIP_PID" 2>/dev/null || true
  stage_now "$([ "$code" = 0 ] && echo finished || echo "exited (status $code)")"
  ship_logs || true
  exit "$code"
}
trap logship_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
