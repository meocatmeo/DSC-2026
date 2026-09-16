#!/usr/bin/env bash
# Phase six: the 2B reranker. The last untested bet with real evidence behind it.
#
#   bash gpu-phase6.sh preflight   # ~20 min  does it load, does it rank, how fast
#   bash gpu-phase6.sh train       # ~3 h     LoRA fine-tune on the existing groups
#   bash gpu-phase6.sh measure     # ~1.5 h   validation + paired comparison vs 0.9493
#   bash gpu-phase6.sh submit      # ~1.5 h   public prediction + packed zip
#   bash gpu-phase6.sh             # all four
#
# It logs itself: rclone cat gdrive:DSC-2026-results/logs/STATUS.txt
#
# THE BET
# -------
# Deployed today: 0.6B reranker + 0.6B embedder = 1.2B of a 4B allowance. Cross-encoder
# reranking follows a power law in parameter count (arXiv 2603.04816), and this is the
# one remaining change that adds capacity rather than re-weighting what is already there.
# Prism-Qwen3.5-Reranker-2B + the embedder is 2.6B, comfortably inside the budget.
#
# WHAT IT COSTS AFTER IT SHIPS
# ----------------------------
# A 2B reranker is ~3x slower than the 0.6B. Every future measurement pays that, and if
# the competition caps inference time it may be disqualifying regardless of accuracy.
# preflight prints the measured throughput so that is a decision, not a surprise.
#
# STOP AT preflight IF IT FAILS. The ranking check there costs twenty minutes; finding
# out after training costs four and a half hours.
set -euo pipefail

echo "gpu-phase6.sh is a historical, unshipped Qwen 2B experiment and is disabled."
echo "Use: bash gpu-phase4.sh multipos"
exit 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
# Exported, not just set: stage_preflight shells out to gpu-preflight-2b.sh, which has
# its own PY default. Without the export, `PY=.venv5/... bash gpu-phase6.sh preflight`
# would run the preflight under the OLD interpreter and report a KeyError on an
# architecture the new one supports perfectly well.
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
LORA_RANK="${LORA_RANK:-16}"        # 0 = full fine-tune; needs ~40 GB for a 2B model
# EPOCHS is a real dial, not a truncation. The cosine schedule is sized from it, so
# EPOCHS=1 is a COMPLETE run that warms up and anneals to zero over one pass. Killing a
# 2-epoch run at the halfway mark is not the same thing at all: it leaves the learning
# rate near its peak, and the saved weights are a mid-training snapshot rather than a
# converged one. If you want half the time, ask for one epoch - do not stop two early.
EPOCHS="${EPOCHS:-2}"
# Unset to keep gradient checkpointing (the safe default). CHECKPOINTING=0 recomputes
# nothing, costs VRAM, and returns about 25% of the wall clock.
CHECKPOINTING="${CHECKPOINTING:-1}"
RERANK_BATCH="${RERANK_BATCH:-16}"  # 2B at 768 tokens; halve if it OOMs
MAXLEN="${MAXLEN:-768}"
export HF_HUB_DISABLE_TELEMETRY=1

LOG="$ROOT/outputs/phase6.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
backup() {
  command -v rclone >/dev/null || return 0
  step "backing up to $RESULTS"
  [ -f models2/reranker-2b-tuned/config.json ] && \
    rclone copy models2/reranker-2b-tuned "$RESULTS/models/reranker-2b-tuned" \
        --transfers 4 2>&1 | tail -1 || true
  rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -1 || true
  step "backup done"
}

DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)
BASE="models2/reranker2b"
TUNED="models2/reranker-2b-tuned"
# The settings behind 0.9493: tuned fusion weights plus the phase-four stacker.
SET() {
  "$PY" - "$1" <<'JSON'
import json, sys
from pathlib import Path
out = json.loads(Path("outputs/v2-fusion-tuning.json").read_text())["best"]
out["stacker"] = "outputs/v4-stacker.json"
out["rerank_batch"] = int(sys.argv[1])
out["rerank_max_length"] = 768
print(json.dumps(out))
JSON
}

# The settings behind 0.9493 live in two files that gpu-restore.sh pulls from Drive.
# If either is missing, every prediction here silently reverts to DEFAULTS and measures
# a different system than the one it claims to be comparing against - so check up front
# rather than discovering it in the diff at the end of a 1.5 h stage.
need_settings() {
  local missing=0
  for f in outputs/v2-fusion-tuning.json outputs/v4-stacker.json; do
    [ -f "$f" ] || { step "MISSING $f"; missing=1; }
  done
  [ "$missing" = 0 ] || {
    step "these carry the tuned fusion weights and the phase-four stacker - without"
    step "them this measures a different pipeline. Restore them:"
    step "  rclone copy $RESULTS/outputs outputs -P"
    return 1
  }
}

stage_preflight() { stage_now "preflight"; bash gpu-preflight-2b.sh; }

stage_train() {
  stage_now "train"
  [ -f "$BASE/config.json" ] || { step "run preflight first - $BASE is not downloaded"; return 1; }
  local groups=artifacts2/groups/train-mp.jsonl
  [ -f "$groups" ] || groups=artifacts2/groups/train-r2.jsonl
  [ -f "$groups" ] || { step "no mined groups found"; return 1; }
  step "training on $groups (already mined - the candidate distribution is unchanged,"
  step "so these negatives are still the ones a 2B model will face at inference)"

  say "LoRA fine-tune the 2B reranker (~3 h)"
  step "rank $LORA_RANK, merged on save, so what ships is an ordinary 2B checkpoint"
  if done_already "$TUNED/config.json"; then
    step "already trained; FORCE=1 to retrain"
  else
    local extra=()
    [ "$CHECKPOINTING" = "0" ] && extra+=(--no-checkpointing)
    step "epochs $EPOCHS, gradient checkpointing $([ "$CHECKPOINTING" = 0 ] && echo off || echo on)"
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --reranker-model "$BASE" --groups "$groups" \
        --lora-rank "$LORA_RANK" --max-length "$MAXLEN" \
        --group-size 16 --groups-per-step 1 --accumulate 8 --epochs "$EPOCHS" \
        "${extra[@]}" --output "$TUNED"
  fi
  backup
  gate "group-size 16 with accumulate 8 keeps 128 pairs per optimiser step at half the
     VRAM of 32x4 - a 2B model needs the headroom. If it OOMed, halve group-size again
     and double accumulate; the effective batch is what matters, not the split."
}

stage_measure() {
  stage_now "measure"
  [ -f "$TUNED/config.json" ] || { step "train first"; return 1; }
  need_settings || return 1
  say "validation with the 2B reranker (~50 min - it is ~3x slower)"
  if done_already outputs/v6-2b.metrics.json; then
    step "already measured"
  else
    timed "predict validation" "$PY" -m legalir2 predict --evaluate \
        "${DENSE[@]}" --reranker-model "$TUNED" --settings "$(SET "$RERANK_BATCH")" \
        --output outputs/v6-2b.json
  fi

  # Which baseline is honest depends on `gpu-venv5.sh baseline`. The 2B runs under
  # transformers 5; v4-stacked was measured under 4.51.3. If those two libraries rank
  # the 0.6B pipeline identically, v4-stacked is a valid comparison point and this is
  # a clean one-variable test. If they do not, comparing against it measures the model
  # change and the library change added together, and the fix is to compare against the
  # 0.6B as re-measured under transformers 5:
  #
  #   BASELINE=outputs/v5-baseline-tf5.diagnostics.json bash gpu-phase6.sh measure
  BASELINE="${BASELINE:-outputs/v4-stacked.diagnostics.json}"
  [ -f "$BASELINE" ] || { step "baseline $BASELINE not found"; return 1; }
  say "against $BASELINE, paired"
  "$PY" -m legalir2 compare --before "$BASELINE" \
      --after outputs/v6-2b.diagnostics.json --label "2B reranker vs baseline"

  say "deployed parameter budget"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$TUNED"
  backup
  gate "submit only if that interval CLEARS ZERO. Four things have now measured as
     nothing at this resolution; a fifth is the likely outcome, and a new submission
     that spans zero spends a slot to learn nothing. The audit total must also be
     under 4,000,000,000."
}

stage_submit() {
  stage_now "submit"
  [ -f outputs/v6-2b.metrics.json ] || { step "measure first"; return 1; }
  need_settings || return 1
  [ -f public-official.json ] || { step "public-official.json missing - rclone copy it"; return 1; }
  say "public test with the 2B reranker (~50 min)"
  timed "predict public" "$PY" -m legalir2 predict --questions public-official.json \
      "${DENSE[@]}" --reranker-model "$TUNED" --settings "$(SET "$RERANK_BATCH")" \
      --output outputs/submission-v6.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v6.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v6.json \
      --output outputs/submission-v6.zip
  ls -lh outputs/submission-v6.zip
  backup
}

case "$STAGE" in
  preflight) stage_preflight ;;
  train)     stage_train ;;
  measure)   stage_measure ;;
  submit)    stage_submit ;;
  all)       stage_preflight; stage_train; stage_measure ;;
  *)         echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
