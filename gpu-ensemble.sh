#!/usr/bin/env bash
# The multi-reranker stacker: the one significant result, un-confounded.
#
#   bash gpu-ensemble.sh            # collect, fit, measure  (~2 h)
#   bash gpu-ensemble.sh collect    # ~55 min  features from BOTH rerankers
#   bash gpu-ensemble.sh fit        # ~1 min   CPU only
#   bash gpu-ensemble.sh measure    # ~30 min  paired against 0.9493
#   bash gpu-ensemble.sh submit     # ~40 min  public prediction + zip
#
# WHY
# ---
# Qwen3-Reranker-0.6B ensembled with reranker-g32 measured +0.0086 [+0.0011, +0.0163],
# p=0.035 - the only interval in ten interventions that excluded zero. Adding the
# stacker then took it to +0.0003 with 1,030 of 1,050 predictions identical.
#
# That was not redundancy. `build_features` read passage_scores[0] and nothing else, so
# the moment a stacker loaded, reranker two was scored, paid for and discarded. The
# ranking it produced was the single-model ranking, which is exactly what 1,030
# identical predictions means.
#
# stack.py now emits a feature block per reranker plus three agreement features, and
# Stacker refuses a weight vector whose width does not match the pipeline rather than
# silently ignoring a model. This stage is that experiment, run properly.
#
# BUDGET: 0.57 embedder + 0.57 g32 + 0.60 Qwen3-Reranker = 1.73B of 4B.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
PARTNER="${PARTNER:-models2/partner_qwen}"
SHIPPED="${SHIPPED:-models2/reranker-g32}"
STACKER="${STACKER:-outputs/v9-stacker-ens.json}"
LIMIT="${LIMIT:-2500}"
DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)

LOG="$ROOT/outputs/ensemble.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
backup() { command -v rclone >/dev/null && rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -1 || true; }

# stage2_candidates = candidates so the partner scores every shortlisted document.
SET() {
  "$PY" - "$@" <<'JSON'
import json, sys
from pathlib import Path
out = json.loads(Path("outputs/v2-fusion-tuning.json").read_text())["best"]
out["stage2_candidates"] = out.get("candidates", 50)
out["stacker"] = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
print(json.dumps(out))
JSON
}

ready() {
  local bad=0
  for d in "$SHIPPED" "$PARTNER"; do
    [ -f "$d/config.json" ] || { step "MISSING $d"; bad=1; }
  done
  [ -f outputs/v2-fusion-tuning.json ] || { step "MISSING outputs/v2-fusion-tuning.json"; bad=1; }
  [ -f outputs/v4-stacked.diagnostics.json ] || { step "MISSING the 0.9493 diagnostics"; bad=1; }
  if [ "$bad" = 1 ] && command -v rclone >/dev/null; then
    step "restoring from Drive"
    [ -f "$PARTNER/config.json" ] || ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        "$PY" -m legalir2 download --which partner_qwen ) || true
    rclone copy "$RESULTS/models/$(basename "$SHIPPED")" "$SHIPPED" -P --transfers 4 2>/dev/null || true
    rclone copy "$RESULTS/outputs" outputs -P 2>/dev/null || true
  fi
  "$PY" -c "
from legalir2.stack import feature_names
n = feature_names(2)
assert len(n) == 20, n
print(f'  stack.py emits {len(n)} features for two rerankers: {n[12:]}')" || {
    step "stack.py has no multi-reranker support - rclone copy gdrive:DSC-2026/legalir2 legalir2"
    return 1; }
}

stage_collect() {
  stage_now "collect"
  ready || return 1
  say "features from BOTH rerankers over $LIMIT training questions (~55 min)"
  step "training questions only. Fitting on the validation split would tune the"
  step "combination on the questions used to judge it."
  if done_already artifacts2/groups/stack-ens.jsonl; then
    step "already collected; FORCE=1 to redo"
  else
    timed "stack collect" "$PY" -m legalir2 stack --limit "$LIMIT" \
        "${DENSE[@]}" --reranker-model "$SHIPPED" "$PARTNER" \
        --settings "$(SET)" \
        --cache artifacts2/groups/stack-ens.jsonl --output "$STACKER"
  fi
  backup
}

stage_fit() {
  stage_now "fit"
  say "refit from the cached features (CPU, ~1 min)"
  timed "stack fit" "$PY" -m legalir2 stack --limit "$LIMIT" \
      "${DENSE[@]}" --reranker-model "$SHIPPED" "$PARTNER" --settings "$(SET)" \
      --cache artifacts2/groups/stack-ens.jsonl --output "$STACKER"
  gate "read the printed weight mass on the second reranker and the agreement features.
     Near zero means the partner adds nothing this pipeline did not already have, and
     the measurement below will say the same thing more expensively. Above ~15% means
     the stacker found the partner useful and the paired test is worth running."
}

stage_measure() {
  stage_now "measure"
  [ -f "$STACKER" ] || { step "fit first"; return 1; }
  say "validation, two rerankers under a stacker that can see both (~30 min)"
  if ! done_already outputs/v9-ensemble.metrics.json; then
    timed "predict validation" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
        --reranker-model "$SHIPPED" "$PARTNER" --settings "$(SET "$STACKER")" \
        --output outputs/v9-ensemble.json
  fi
  say "against the 0.9493 configuration, paired"
  "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/v9-ensemble.diagnostics.json --label "two-reranker stacker vs 0.9493"
  say "deployed parameter budget"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$SHIPPED" "$PARTNER"
  backup
  gate "the number to beat on the leaderboard is 0.9591, and you are at 0.9493 with
     3rd-6th place inside 0.9546-0.9559. A validation gain of +0.006 would put this
     configuration in that cluster. Submit if the interval clears zero; submit anyway
     if it is positive and large, because submissions are unlimited and ranked on
     recall alone - but do NOT read a leaderboard gap under 0.015 as evidence."
}

stage_submit() {
  stage_now "submit"
  [ -f outputs/v9-ensemble.metrics.json ] || { step "measure first"; return 1; }
  say "public test (~40 min - two rerankers is ~2.5x the single-model pass)"
  timed "predict public" "$PY" -m legalir2 predict --questions public-official.json \
      "${DENSE[@]}" --reranker-model "$SHIPPED" "$PARTNER" \
      --settings "$(SET "$STACKER")" --output outputs/submission-v9.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v9.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v9.json \
      --output outputs/submission-v9.zip
  ls -lh outputs/submission-v9.zip
  command -v rclone >/dev/null && rclone copy outputs/submission-v9.zip "$RESULTS/outputs" 2>&1 | tail -1
  backup
}

case "$STAGE" in
  collect) stage_collect ;;
  fit)     stage_fit ;;
  measure) stage_measure ;;
  submit)  stage_submit ;;
  all)     stage_collect; stage_measure ;;
  *)       echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
