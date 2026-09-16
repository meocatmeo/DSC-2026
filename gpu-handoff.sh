#!/usr/bin/env bash
# Close out legalir2, then leave the box ready for legalir3 - without terminating it.
#
#   bash gpu-handoff.sh            # save, check, ready - then prints what to run
#   bash gpu-handoff.sh save       # push everything irreplaceable to Drive
#   bash gpu-handoff.sh check      # was a tuned embedder already tried and lost?
#   bash gpu-handoff.sh ready      # are legalir3's prerequisites all present?
#
# `save` first, always. Everything after it is cheap to redo; the trained checkpoints
# and the diagnostics are not, and a rented box can disappear between two commands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '  [ok]      %s\n' "$*"; }
miss() { printf '  [MISSING] %s\n' "$*"; }

stage_save() {
  say "pushing everything irreplaceable to $RESULTS"
  command -v rclone >/dev/null || { step "rclone not installed - nothing can be saved"; return 1; }

  # Small and irreplaceable: every metric, diagnostic, stacker, settings file, log and
  # submission zip this project has produced. A few MB, and it is the entire audit trail.
  step "outputs (metrics, diagnostics, stackers, logs, submissions)"
  rclone copy outputs "$RESULTS/outputs" -P --transfers 8

  # Mined groups: 23-30 minutes of GPU each, and every future training run starts here.
  step "mined groups"
  rclone copy artifacts2/groups "$RESULTS/groups" -P 2>/dev/null || true

  # Trained checkpoints: hours each. Skipped if a stage never ran.
  for name in reranker-final reranker-doc-base reranker-doc embedding-l3; do
    if [ -f "models2/$name/config.json" ]; then
      step "models2/$name ($(du -sh "models2/$name" 2>/dev/null | cut -f1))"
      rclone copy "models2/$name" "$RESULTS/models/$name" -P --transfers 4
    fi
  done

  # A rebuilt dense index is 1.2 h of GPU and is only valid for the encoder that made it.
  for d in dense-l3; do
    [ -f "artifacts2/$d/manifest.json" ] && {
      step "artifacts2/$d"
      rclone copy "artifacts2/$d" "$RESULTS/$d" -P --transfers 4; }
  done

  say "what is now in Drive"
  rclone size "$RESULTS"
  printf '\n  submissions:\n'
  rclone lsf "$RESULTS/outputs" | grep -i '\.zip$' | sed 's/^/    /' || echo "    none"
  printf '\n  checkpoints:\n'
  rclone lsf "$RESULTS/models/" | sed 's/^/    /'
}

stage_check() {
  say "was a tuned embedder already tried, and did it lose?"
  step "gpu-restore.sh expects an 'embedding-tuned' directory, which means a bi-encoder"
  step "was trained at some point in this project and never shipped. If it was measured"
  step "and lost, legalir3's first stage is a weaker bet than it looks - and this costs"
  step "two minutes to find out rather than four and a half hours."
  command -v rclone >/dev/null || { step "no rclone"; return 0; }

  printf '\n  embedder checkpoints in Drive:\n'
  rclone lsf "$RESULTS/models/" 2>/dev/null | grep -i embed | sed 's/^/    /' \
    || echo "    none - no bi-encoder was ever trained"

  printf '\n  dense indexes in Drive:\n'
  rclone lsf "$RESULTS/" 2>/dev/null | grep -i dense | sed 's/^/    /' || echo "    none"

  printf '\n  any measurement of one:\n'
  local found
  found="$(rclone lsf "$RESULTS/outputs" 2>/dev/null \
           | grep -iE 'dense|embed|bi-' | grep -i 'metrics' || true)"
  if [ -n "$found" ]; then
    echo "$found" | sed 's/^/    /'
    printf '\n  recall in each:\n'
    for f in $found; do
      rclone cat "$RESULTS/outputs/$f" 2>/dev/null | "$PY" -c "
import json,sys
try:
    m = json.load(sys.stdin)
    c = m.get('candidate_recall', {})
    print(f\"    $f  Recall@5 {m.get('recall', float('nan')):.4f}  \"
          f\"candidate @5 {c.get('@5','-')} @50 {c.get('@50','-')}\")
except Exception as error:
    print(f'    $f  (unreadable: {error})')"
    done
    printf '\n  Compare those against 0.9371 (reranker only, no stacker) and the\n'
    printf '  0.8914/0.9805 candidate recall baseline. A tuned embedder that already\n'
    printf '  measured LOWER was trained on fused-distribution negatives - which is the\n'
    printf '  thing legalir3 changes, so it is evidence about the old recipe, not the new\n'
    printf '  one. A tuned embedder that measured HIGHER and was never shipped is a\n'
    printf '  free win sitting in Drive.\n'
  else
    echo "    none found - it was trained but apparently never measured"
    printf '\n  That is the best case for legalir3: the bet has not been tested, and it\n'
    printf '  would have been tested with the wrong negatives anyway.\n'
  fi
}

stage_ready() {
  say "legalir3 prerequisites on this box"
  local bad=0
  check_file() { [ -e "$1" ] && ok "$2" || { miss "$2  ($1)"; bad=1; }; }

  check_file artifacts2/store.sqlite            "chunk store"
  check_file artifacts2/lexical/bigram.npz      "lexical postings"
  check_file artifacts2/dense/manifest.json     "current dense index (the baseline)"
  check_file models2/embedding/config.json      "base embedder - what train-bi starts from"
  check_file outputs/v2-fusion-tuning.json      "tuned fusion weights (every stage reads these)"
  check_file outputs/v4-stacked.diagnostics.json "0.9493 diagnostics (endtoend compares here)"
  check_file legalir2/retrieval.py              "candidate-recall measurement"
  check_file gpu-legalir3.sh                    "the phase script"

  printf '\n'
  if grep -q '"dense"' legalir2/mine.py 2>/dev/null && \
     grep -q 'channel' legalir2/mine.py 2>/dev/null; then
    ok "mine.py understands --channel"
  else
    miss "mine.py has no --channel - rclone copy gdrive:DSC-2026/legalir2 legalir2"; bad=1
  fi

  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -ge 12 ]; then
    ok "disk: ${free} GB free (needs ~8 for a second index and encoder)"
  else
    miss "disk: only ${free} GB free - a second dense index needs ~8"; bad=1
  fi

  printf '\n'
  if [ "$bad" = 0 ]; then
    cat <<'NEXT'
  Everything legalir3 needs is here.

      bash gpu-legalir3.sh

  ~4.5 h: baseline, mine on the dense channel, train the bi-encoder, rebuild the
  index, then a PAIRED candidate-recall comparison.

  Read the baseline first. Fused @5 must be 0.8914 and @50 0.9805. If they are not,
  the settings do not match the shipped system and every later comparison is against
  the wrong thing - stop there rather than spending four hours on it.
NEXT
  else
    printf '  Fix the [MISSING] lines above before starting legalir3.\n'
    return 1
  fi
}

case "$STAGE" in
  save)  stage_save ;;
  check) stage_check ;;
  ready) stage_ready ;;
  all)   stage_save; stage_check; stage_ready ;;
  *)     echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
