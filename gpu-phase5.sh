#!/usr/bin/env bash
# Phase five: build the measurement harness, once, so every later experiment is free.
#
#   bash gpu-phase5.sh crossfit   # ~2 h   two rerankers, each blind to half the data
#   bash gpu-phase5.sh cache      # ~3.5 h freeze candidates + scores for all ~7,000
#   bash gpu-phase5.sh replay     # ~0 h   search every setting, CPU only
#   bash gpu-phase5.sh final      # ~1 h   validate, predict the public test, pack
#   bash gpu-phase5.sh            # all four, ending in a submission zip
#
# It logs itself. From any machine, at any time:
#   rclone cat gdrive:DSC-2026-results/logs/STATUS.txt   # heartbeat + last 40 lines
#   rclone cat gdrive:DSC-2026-results/logs/phase5.log   # everything
#
# WHY THIS EXISTS
# ---------------
# Phase four spent ~9 GPU hours and returned one readable result from four experiments.
# The other three were not wrong, they were unmeasurable: a 1,050-question holdout
# resolves a paired difference of about 0.0094, and every remaining improvement is in
# the 0.003-0.008 band. Scoring across all ~7,000 labelled questions makes that 0.0036.
#
# WHY CROSS-FITTING IS NOT OPTIONAL
# ---------------------------------
# reranker-g32 was trained on 5,950 of the 7,000. Scoring those with it measures what it
# memorised, and a bigger cache built that way is just a more confident way to be wrong.
# So: split the questions in two, train a reranker on each half, and score every question
# with the model that never saw it.
#
# WHAT IT COSTS, HONESTLY
# -----------------------
# ~5.5 GPU hours in total - more than the ~3 h I estimated when proposing this, because
# two trainings plus two deep scoring passes is simply more work than I accounted for.
# It is spent once. After it, tuning the stacker, the aggregation, the candidate depth
# and the blend weights costs seconds of CPU instead of 16 GPU minutes each.
# Set DEPTH=50 to halve the caching time at the cost of not being able to test
# candidates beyond 50.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups artifacts2/cache
PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
DEPTH="${DEPTH:-100}"            # documents per question the reranker scores
PASSAGES="${PASSAGES:-3}"        # passages per document; caps what the sweep can test
ALL_QUESTIONS="${ALL_QUESTIONS:-train.json}"   # all 7,000 labelled questions
# Passages scored per forward pass. The default of 8 was set for a machine nobody
# specified and leaves any modern card mostly idle: 8 x 768 tokens through a 0.6B model
# is a rounding error on 32 GB. Batching changes no score - each passage is scored
# independently - so this is throughput for free. 32 is safe on 24 GB, 64 on 32 GB+.
RERANK_BATCH="${RERANK_BATCH:-32}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Self-logging: every line captured, a heartbeat shipped to Drive every 60 s, and a
# final copy on exit even after Ctrl-C. No second terminal, no side loop to remember.
LOG="$ROOT/outputs/phase5.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
# This script produces ~6 hours of artifacts and has no autopilot behind it, so it
# backs itself up between stages. The two cross-fitted checkpoints and the score
# caches are the expensive outputs; losing them to a terminated rental would mean
# paying for the whole harness twice.
backup() {
  command -v rclone >/dev/null || return 0
  step "backing up to $RESULTS"
  for d in models2/reranker-f0 models2/reranker-f1; do
    [ -f "$d/config.json" ] && rclone copy "$d" "$RESULTS/models/$(basename "$d")" \
        --transfers 4 2>&1 | tail -1 || true
  done
  rclone copy artifacts2/cache  "$RESULTS/cache"  --transfers 4 2>&1 | tail -1 || true
  rclone copy artifacts2/groups "$RESULTS/groups" 2>&1 | tail -1 || true
  rclone copy outputs           "$RESULTS/outputs" 2>&1 | tail -1 || true
  step "backup done"
}

DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)
FUSION="outputs/v2-fusion-tuning.json"
settings() {
  "$PY" - "$FUSION" "$RERANK_BATCH" <<'JSON'
import json, sys
from pathlib import Path
out = {}
path = Path(sys.argv[1])
if path.is_file():
    out.update(json.loads(path.read_text())["best"])
out["rerank_batch"] = int(sys.argv[2])
print(json.dumps(out))
JSON
}

# --------------------------------------------------------- 1. two blind rerankers
stage_crossfit() {
  stage_now "crossfit"
  say "split all questions into two folds and train a reranker on each"
  step "fold assignment is hash based, so a question keeps its fold even if the"
  step "question file is reordered or extended"
  for fold in 0 1; do
    local groups="artifacts2/groups/fold$fold.jsonl"
    local model="models2/reranker-f$fold"
    if done_already "$groups"; then
      step "fold $fold already mined"
    else
      timed "mine fold $fold" "$PY" -m legalir2 mine --negatives 31 \
          --questions "$ALL_QUESTIONS" --fold "$fold" \
          --settings "$(settings)" --output "$groups"
    fi
    if done_already "$model/config.json"; then
      step "fold $fold reranker already trained"
    else
      timed "train fold $fold (~50 min, half the data)" "$PY" -m legalir2 train-ce \
          --groups "$groups" --group-size 32 --groups-per-step 1 --accumulate 4 \
          --epochs 2 --output "$model"
    fi
  done
  gate "two checkpoints, each blind to half the questions. Neither of these ships - they
     exist only so the cache measures generalisation rather than memory. The model you
     deploy is still trained on everything."
}

# ------------------------------------------------------------ 2. freeze the scores
stage_cache() {
  stage_now "cache"
  say "score each fold with the reranker that never saw it (~1.75 h per fold)"
  step "depth $DEPTH documents x $PASSAGES passages per question, batch $RERANK_BATCH"
  for fold in 0 1; do
    local other=$(( 1 - fold ))
    local out="artifacts2/cache/fold$fold.npz"
    if done_already "$out"; then
      step "fold $fold already cached"
      continue
    fi
    [ -f "models2/reranker-f$other/config.json" ] || {
      step "missing models2/reranker-f$other - run crossfit first"; return 1; }
    timed "cache fold $fold with reranker-f$other" "$PY" -m legalir2 cache \
        --questions "$ALL_QUESTIONS" --fold "$fold" \
        "${DENSE[@]}" --reranker-model "models2/reranker-f$other" \
        --settings "$(settings)" --depth 200 --rerank-depth "$DEPTH" \
        --cache-passages "$PASSAGES" --output "$out"
  done
  ls -lh artifacts2/cache/
  gate "from here on, nothing below needs a GPU. If a later experiment wants a setting
     outside what was cached - candidates above $DEPTH, or passages above $PASSAGES -
     it needs a new caching pass, so check that before assuming an idea is cheap."
}

# --------------------------------------------- 3. re-decide what phase four could not
stage_replay() {
  stage_now "replay"
  local caches=(artifacts2/cache/fold0.npz artifacts2/cache/fold1.npz)
  for c in "${caches[@]}"; do
    [ -f "$c" ] || { step "missing $c - run cache first"; return 1; }
  done
  local base; base="$(settings)"

  say "baseline: what shipped, measured at the new resolution"
  "$PY" -m legalir2 cv --cache "${caches[@]}" --settings "$base" --label "shipped defaults"

  say "fit the learned rank fusion across both folds"
  step "fitted on cross-fitted scores, so it can no longer lean on the reranker having"
  step "memorised its training questions - which the phase-four stacker could"
  "$PY" -m legalir2 cv --cache "${caches[@]}" --settings "$base" \
      --fit-stacker outputs/v5-stacker.json \
      --against "$base" --label "learned fusion (cross-fitted)"

  say "search A: the hand-weighted blend, every knob, no stacker"
  step "this is the fair opponent. Comparing the stacker against UNTUNED defaults"
  step "flatters it - the honest question is whether it beats the best blend there is."
  "$PY" -m legalir2 cv --cache "${caches[@]}" --settings "$base" --search \
      --write-settings outputs/v5-settings-blend.json

  say "search B: with the stacker"
  step "a stacker replaces aggregate / passages / rerank_blend entirely, so only"
  step "candidates is still live - the rest are skipped rather than scored as +0.0000."
  "$PY" -m legalir2 cv --cache "${caches[@]}" --settings "$base" \
      --stacker outputs/v5-stacker.json --search \
      --write-settings outputs/v5-settings.json

  say "A versus B, paired, at full resolution"
  "$PY" -m legalir2 cv --cache "${caches[@]}" \
      --settings "$("$PY" -c "
import json;print(json.dumps(json.load(open('outputs/v5-settings.json'))['best']))")" \
      --stacker outputs/v5-stacker.json \
      --against "$("$PY" -c "
import json;print(json.dumps(json.load(open('outputs/v5-settings-blend.json'))['best']))")" \
      --label "tuned stacker vs tuned blend"

  # Whichever wins is what stage_final deploys.
  "$PY" - <<'PICK'
import json
from pathlib import Path
blend = json.loads(Path("outputs/v5-settings-blend.json").read_text())
stack = json.loads(Path("outputs/v5-settings.json").read_text())
print(f"\n  tuned blend   Recall@5 {blend['recall']:.4f}")
print(f"  tuned stacker Recall@5 {stack['recall']:.4f}")
if blend["recall"] > stack["recall"]:
    Path("outputs/v5-settings.json").write_text(json.dumps(blend, indent=1))
    print("  the blend won - stage_final will deploy it, stacker dropped")
else:
    print("  the stacker won - stage_final will deploy it")
PICK
  gate "every ADOPT line is a measured gain at 0.0036 resolution. A dash means the
     change is worth less than that - which is also below what the public leaderboard
     can distinguish, so it is not worth a submission slot."
}

# ------------------------------------------------- 4. build and pack the submission
stage_final() {
  stage_now "final"
  [ -f outputs/v5-settings.json ] || { step "run replay first"; return 1; }
  local chosen; chosen="$("$PY" -c "
import json; print(json.dumps(json.load(open('outputs/v5-settings.json'))['best']))")"
  say "chosen configuration"
  echo "  $chosen"

  # The deployed reranker is still the one trained on everything. The cross-fitted
  # models exist only to make the cache honest; each saw half the data and would
  # ship a weaker system.
  local model="models2/reranker-g32"
  say "validate the chosen settings on the DEPLOYED reranker (~25 min)"
  step "the stacker was fitted against cross-fitted scores and is applied here to a"
  step "different checkpoint. Features are per-question normalised so it should"
  step "transfer - but should is not does, and this run is the check."
  if done_already outputs/v5-final.metrics.json; then
    step "already measured"
  else
    timed "predict validation" "$PY" -m legalir2 predict --evaluate \
        "${DENSE[@]}" --reranker-model "$model" --settings "$chosen" \
        --output outputs/v5-final.json
  fi
  [ -f outputs/v4-stacked.diagnostics.json ] && \
    "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
        --after outputs/v5-final.diagnostics.json \
        --label "v5 settings vs the 0.9493 submission"

  say "predict the public test and pack"
  timed "predict public" "$PY" -m legalir2 predict \
      --questions public-official.json \
      "${DENSE[@]}" --reranker-model "$model" --settings "$chosen" \
      --output outputs/submission-v5.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v5.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v5.json \
      --output outputs/submission-v5.zip

  say "deployed parameter budget"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$model"
  ls -lh outputs/submission-v5.zip
  gate "submit ONLY if the paired comparison above cleared zero. If it spans zero the
     v5 settings are not measurably better than what produced 0.9493, and a new
     submission spends a slot to learn nothing."
}

case "$STAGE" in
  crossfit) stage_crossfit ;;
  cache)    stage_cache ;;
  replay)   stage_replay ;;
  final)    stage_final; backup ;;
  all)      stage_crossfit; backup; stage_cache; backup; stage_replay; stage_final; backup ;;
  *)        echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
