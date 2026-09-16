#!/usr/bin/env bash
# Unattended: everything from here to a packed submission, with no one watching.
#
#   tmux new -s auto
#   bash gpu-auto.sh
#   ctrl-b d      then close the laptop
#
# Runs multipos -> stack -> sweep -> ceiling, backs each finished stage up to Drive
# as it completes, then picks the best validated configuration *by its own recorded
# manifest* and packs a submission from it.
#
# Watch it from anywhere, including a phone:
#
#   rclone cat gdrive:DSC-2026-results/logs/STATUS.txt     # heartbeat + last 40 lines
#   rclone cat gdrive:DSC-2026-results/logs/auto.log       # the whole thing
#
# Three deliberate differences from gpu-phase2.sh, all because nobody is watching:
#
#   * Every line of output - this script, the stages, python, tracebacks - is captured
#     and shipped to Drive every few minutes, not only when a stage ends. A two hour
#     training stage would otherwise be two hours of silence from outside the box.
#   * A failing stage does not kill the run. Later stages that depend on it are
#     skipped, everything already earned is still uploaded, and a submission is still
#     packed from the best model that did finish. Coming back to a dead box with no
#     submission because step three of five raised is the outcome this avoids.
#   * Every predict is registered with the checkpoints that produced it, so the final
#     choice is made from recorded provenance rather than from a filename convention
#     that someone has to remember is still true.
#
# Safe to re-run: finished stages are detected and skipped. FORCE=1 redoes them.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
PUBLIC="${PUBLIC:-public-official.json}"
LOG_EVERY="${LOG_EVERY:-60}"           # seconds between log uploads. A minute is
                                       # cheap - a heartbeat is a few KB - and it is
                                       # the difference between "is it alive?" being
                                       # answerable now or in five minutes.
mkdir -p outputs artifacts2/groups
LOG="$ROOT/outputs/auto.log"
STATE="$ROOT/outputs/.auto-state"
STARTED_AT=$(date +%s)

# Everything written from here on - by this script, by the stages it calls, by python,
# by a traceback nobody anticipated - lands in the log. Doing this once at the top is
# what makes the promise "all of it" true; per-command `| tee` only ever captures the
# commands someone remembered to pipe.
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

STATUS_OK=""; STATUS_BAD=""; STAGE_NOW="starting"; SHIP_FAILURES=0
# The log shipper runs in a background subshell forked before most of this happens, so
# it cannot see these variables. State goes through a file instead, or the heartbeat
# would cheerfully report the run's opening moments for the next eight hours.
save_state() {
  printf 'stage=%s\nok=%s\nbad=%s\nstarted=%s\n' \
      "$STAGE_NOW" "${STATUS_OK:- none}" "${STATUS_BAD:- none}" "$STARTED_AT" > "$STATE"
}
save_state

# ------------------------------------------------------------------ log shipping
ship_logs() {
  command -v rclone >/dev/null || return 0
  local stage ok bad started elapsed
  stage="?"; ok="?"; bad="?"; started="$STARTED_AT"
  if [ -f "$STATE" ]; then
    stage=$(sed -n 's/^stage=//p' "$STATE"); ok=$(sed -n 's/^ok=//p' "$STATE")
    bad=$(sed -n 's/^bad=//p' "$STATE");     started=$(sed -n 's/^started=//p' "$STATE")
  fi
  elapsed=$(( ($(date +%s) - started) / 60 ))
  {
    printf 'DSC-2026 autopilot\n'
    printf 'host            : %s\n' "$(hostname)"
    printf 'heartbeat (UTC) : %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'running for     : %s h %s min\n' "$((elapsed/60))" "$((elapsed%60))"
    printf 'current stage   : %s\n' "$stage"
    printf 'stages done     : %s\n' "$ok"
    printf 'stages failed   : %s\n' "$bad"
    if command -v nvidia-smi >/dev/null; then
      printf 'gpu             : %s\n' \
        "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
             --format=csv,noheader 2>/dev/null | head -1)"
    fi
    printf 'disk free       : %s\n' "$(df -h "$ROOT" | awk 'NR==2{print $4}')"
    printf '\n--- last 40 lines of auto.log ---\n'
    tail -n 40 "$LOG" 2>/dev/null
  } > "$ROOT/outputs/STATUS.txt" 2>/dev/null
  # Logs only - the big artefacts travel in backup(), which runs far less often.
  #
  # Errors are recorded, never swallowed. An earlier version sent rclone's stderr to
  # /dev/null "to keep the log clean", which meant the one component whose job is
  # visibility was the only one that could fail in complete silence - no logs in Drive
  # and no way to find out why. Failures go to a file the next successful upload
  # carries with it, and a repeated failure prints into the run log itself.
  local report="$ROOT/outputs/rclone-errors.log"
  # --ignore-checksum because auto.log is being appended to WHILE it uploads, so its
  # md5 legitimately differs between read and verify. rclone calls that "corrupted on
  # transfer", retries, and succeeds - correct behaviour, but it buries the real errors
  # this file exists to surface under nine hours of false alarms.
  if ! rclone copy "$ROOT/outputs" "$RESULTS/logs" --ignore-checksum \
          --include "*.log" --include "STATUS.txt" --include "*.stats.json" \
          >>"$report" 2>&1; then
    SHIP_FAILURES=$((SHIP_FAILURES + 1))
    printf '[%s] log upload FAILED (%s in a row) - see outputs/rclone-errors.log\n' \
        "$(date +%H:%M:%S)" "$SHIP_FAILURES" >> "$LOG"
    if [ "$SHIP_FAILURES" = 1 ] || [ $((SHIP_FAILURES % 12)) = 0 ]; then
      printf '  last rclone error:\n' >> "$LOG"
      tail -n 6 "$report" | sed 's/^/    /' >> "$LOG"
    fi
    return 1
  fi
  [ "$SHIP_FAILURES" -gt 0 ] && printf '[%s] log upload recovered\n' \
      "$(date +%H:%M:%S)" >> "$LOG"
  SHIP_FAILURES=0
  rclone copy "$ROOT" "$RESULTS/logs" --ignore-checksum --include "*.log" --max-depth 1 >>"$report" 2>&1 || true
  return 0
}

# Before anything expensive, prove the log channel actually works. Discovering at
# hour three that nothing has been shipped is the failure this exists to prevent.
if command -v rclone >/dev/null; then
  printf '[%s] testing the log channel to %s ...\n' "$(date +%H:%M:%S)" "$RESULTS/logs"
  if ship_logs; then
    printf '[%s] log channel OK - STATUS.txt is in %s/logs\n' "$(date +%H:%M:%S)" "$RESULTS"
  else
    printf '\n  WARNING: cannot upload logs to %s/logs.\n' "$RESULTS"
    printf '  The run continues and everything is still written to %s locally,\n' "$LOG"
    printf '  but you will not be able to watch it from off the box. Error above.\n\n'
  fi
else
  printf '\n  WARNING: rclone is not installed - nothing will be uploaded at all.\n\n'
fi

( while sleep "$LOG_EVERY"; do ship_logs; done ) &
SHIPPER=$!
cleanup() {
  kill "$SHIPPER" 2>/dev/null
  STAGE_NOW="exited"; save_state
  ship_logs
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- Drive backup
# Called after every stage rather than once at the end. Uploading 2.3 GB twice costs
# minutes; losing a finished checkpoint to a terminated rental costs hours, and this
# box has already been lost once.
backup() {
  command -v rclone >/dev/null || { step "rclone missing - skipping backup"; return 0; }
  step "backing up weights and outputs to $RESULTS"
  for d in models2/reranker-doc models2/reranker-tuned-idx models2/reranker-g32 \
           models2/embedding-tuned; do
    [ -f "$d/config.json" ] && rclone copy "$d" "$RESULTS/models/$(basename "$d")" \
        --transfers 4 2>&1 | tail -2 || true
  done
  # The dense indexes belong here too. They look like derived data - rebuildable from
  # the corpus and a checkpoint - but rebuilding one is 1.2 h of GPU time, which makes
  # them the second most expensive thing on the box after the weights. ~700 MB each,
  # and rclone skips them once uploaded, so this costs one transfer, not one per stage.
  for d in artifacts2/dense artifacts2/dense-tuned; do
    [ -f "$d/manifest.json" ] && rclone copy "$d" "$RESULTS/$(basename "$d")" \
        --transfers 4 2>&1 | tail -2 || true
  done
  rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -2 || true
  rclone copy artifacts2/groups "$RESULTS/groups" 2>&1 | tail -2 || true
  ship_logs
  step "backup done"
}

# ---------------------------------------------------------------- stage runner
run_stage() {
  local name="$1" started=$SECONDS
  STAGE_NOW="$name"; save_state
  say "stage: $name"
  ship_logs
  if bash gpu-phase4.sh "$name"; then
    step "stage $name finished in $(( (SECONDS-started)/60 )) min"
    STATUS_OK="$STATUS_OK $name"; save_state
    backup
    return 0
  fi
  step "STAGE $name FAILED after $(( (SECONDS-started)/60 )) min - continuing without it"
  STATUS_BAD="$STATUS_BAD $name"; save_state
  backup
  return 1
}

# ------------------------------------------------------------------- the run
say "autopilot starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
step "log: $LOG   shipped to $RESULTS/logs every ${LOG_EVERY}s"
step "watch from anywhere: rclone cat $RESULTS/logs/STATUS.txt"

# Ordered by evidence. multipos is the biggest measured pool of recoverable points, so
# it goes first and gets the retry; the mined groups survive a failed training run, so
# only the checkpoint is cleared.
if ! run_stage multipos; then
  step "retrying multipos once - the mined groups are kept, only the checkpoint is redone"
  rm -rf models2/reranker-doc
  run_stage multipos || true
fi

# The stacker fits against whichever reranker exists, so it is useful even if the
# multi-positive retrain failed entirely.
run_stage stack || true
run_stage sweep || true

# The speculative one, last: phase two measured three regressions from improving
# retrieval, and this stage only differs by re-mining through the better index.
run_stage ceiling || true
run_stage audit || true

# --------------------------------------------------- choose, predict, package
STAGE_NOW="choosing a submission"; save_state
say "choosing the configuration to submit"
CHOICE=outputs/auto-choice.json
"$PY" - "$CHOICE" <<'PICK'
"""Pick the best validated run from its own recorded manifest.

Every metrics file has a sibling .run.json holding the settings and the checkpoint
paths the pipeline actually loaded. Reading the choice back out of those two files
means the submission is rebuilt from what was measured, not from a filename that
someone has to remember still maps to a particular recipe.
"""
import glob, json, sys
from pathlib import Path

rows = []
for metrics_path in sorted(glob.glob("outputs/v[24]-*.metrics.json")):
    run_path = Path(metrics_path.replace(".metrics.json", ".run.json"))
    if not run_path.is_file():
        continue
    try:
        metrics = json.loads(Path(metrics_path).read_text())
        run = json.loads(run_path.read_text())
    except Exception as error:
        print(f"  skipping {metrics_path}: {error}")
        continue
    models = run.get("models") or {}
    if not models.get("rerankers"):
        # A lexical-only or retrieval-only ablation. Real, but not a submission.
        print(f"  {Path(metrics_path).name:<34} {metrics['recall']:.4f}  (no reranker, skipped)")
        continue
    rows.append({"name": Path(metrics_path).name.replace(".metrics.json", ""),
                 "recall": float(metrics["recall"]),
                 "candidate_recall": metrics.get("candidate_recall", {}),
                 "settings": run.get("settings", {}),
                 "models": models,
                 "parameters": (run.get("audit") or {}).get("total")})

if not rows:
    sys.exit("\n  NOTHING TO SUBMIT: no validated run recorded a reranker.\n")

rows.sort(key=lambda r: -r["recall"])
print(f"\n  {'run':<22}{'Recall@5':>10}{'cand@100':>10}  models")
for row in rows:
    names = ", ".join(Path(p).name for p in row["models"]["rerankers"])
    if row["models"].get("dense"):
        names += f" + {Path(row['models']['dense']).name}"
    print(f"  {row['name']:<22}{row['recall']:>10.4f}"
          f"{str(row['candidate_recall'].get('@100','-')):>10}  {names}")

best = rows[0]
budget = 4_000_000_000
if best.get("parameters") and best["parameters"] > budget:
    print(f"\n  {best['name']} deploys {best['parameters']:,} parameters, over the "
          f"{budget:,} limit - falling back to the best run inside it")
    inside = [r for r in rows if not r.get("parameters") or r["parameters"] <= budget]
    if not inside:
        sys.exit("\n  NOTHING TO SUBMIT: every run exceeds the parameter budget.\n")
    best = inside[0]

Path(sys.argv[1]).write_text(json.dumps(best, indent=1, ensure_ascii=False))
print(f"\n  chosen: {best['name']}  Recall@5 {best['recall']:.4f}")
if best["recall"] < 0.9371:
    print(f"  NOTE: this is below the 0.9371 validation score behind your 0.9433 submission. The zip is "
          f"still packed, but compare before you upload it.")
PICK

if [ ! -f "$CHOICE" ]; then
  say "no configuration could be chosen - stopping before the public run"
  step "read $LOG, then run 'bash gpu-phase2.sh final' by hand"
  exit 1
fi

STAGE_NOW="public prediction"; save_state
say "predicting the public test with the chosen configuration"
mapfile -t ARGS < <("$PY" - "$CHOICE" <<'ARGS'
import json, sys
best = json.load(open(sys.argv[1]))
models = best["models"]
print("--reranker-model")
for path in models["rerankers"]:
    print(path)
if models.get("dense"):
    print("--dense"); print(models["dense"])
    print("--embedding-model"); print(models["embedding"])
else:
    # The chosen run scored without a dense channel. predict() would otherwise pick
    # the index up from the default path and quietly submit a different system than
    # the one that was measured.
    print("--no-dense")
print("--settings"); print(json.dumps(best["settings"]))
ARGS
)
step "public run arguments: ${ARGS[*]}"

SUB=outputs/submission-v4
if "$PY" -m legalir2 predict --questions "$PUBLIC" "${ARGS[@]}" --output "$SUB.json"; then
  "$PY" -m legalir2 validate --predictions "$SUB.json" \
    && "$PY" -m legalir2 pack --predictions "$SUB.json" --output "$SUB.zip"
else
  step "PUBLIC PREDICTION FAILED - the validated checkpoints are still backed up"
fi

# ------------------------------------------------------------------- wrap up
STAGE_NOW="finished"; save_state
backup
[ -f "$SUB.zip" ] && rclone copy "$SUB.zip" "$RESULTS/outputs" 2>&1 | tail -2 || true

say "autopilot finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  stages completed :%s\n' "${STATUS_OK:- none}"
printf '  stages failed    :%s\n' "${STATUS_BAD:- none}"
if [ -f "$SUB.zip" ]; then
  printf '  submission       : %s\n' "$(ls -lh "$SUB.zip" | awk '{print $5, $9}')"
  "$PY" -c "import json;b=json.load(open('$CHOICE'));print(f'  validation       : Recall@5 {b[\"recall\"]:.4f} from {b[\"name\"]}')"
else
  printf '  submission       : NOT PACKED - read %s\n' "$LOG"
fi
printf '  everything is in %s\n' "$RESULTS"
printf '\n  Before uploading to the leaderboard, check the run table above: submit the\n'
printf '  chosen row only if it beats the 0.9371 behind your 0.9433 submission.\n\n'

# The trap ships the final log on the way out, after these closing lines are written.
