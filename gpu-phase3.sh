#!/usr/bin/env bash
# Phase three: 0.9433 -> the 0.96 attempt. Two moves, cheapest first.
#
#   bash gpu-phase3.sh sweep    # ~30 min  tune the reranking stage. No training.
#   bash gpu-phase3.sh distil   # ~6 h     Qwen3-Reranker-4B teacher -> our student
#   bash gpu-phase3.sh          # both, in order
#
# Assumes a box restored by gpu-restore.sh: reranker-g32, the dense indexes and the
# mined groups all present. Run it under gpu-auto3.sh for unattended operation.
#
# On the parameter budget: the teacher is Qwen3-Reranker-4B and is run ONLY to write a
# score cache. It is never loaded at prediction time and is not part of any submission.
# The model that ships is the student, which is the same 560M cross encoder as before,
# so the deployed count does not change. `audit` prints it at the end so this is a
# measured claim rather than an assurance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
export HF_HUB_DISABLE_TELEMETRY=1

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

BASE_FUSION="outputs/v2-fusion-tuning.json"
TUNED_FUSION="outputs/v2-fusion-tuning-embedder.json"
RERANK_TUNING="outputs/v3-rerank-tuning.json"

# The best reranker and matching dense index from phase two, whichever exists.
BEST_RERANKER="models2/reranker-g32"
[ -f "$BEST_RERANKER/config.json" ] || BEST_RERANKER="models2/reranker-tuned"
if [ -f artifacts2/dense-tuned/manifest.json ]; then
  DENSE_ARGS=(--dense artifacts2/dense-tuned --embedding-model models2/embedding-tuned)
  FUSION_FILE="$TUNED_FUSION"
else
  DENSE_ARGS=(--dense artifacts2/dense --embedding-model models2/embedding)
  FUSION_FILE="$BASE_FUSION"
fi

# Pipeline settings: the tuned fusion weights, with the tuned reranking knobs folded in
# once the sweep has produced them. One function so no stage can disagree with another.
settings() {
  "$PY" - "$FUSION_FILE" "$RERANK_TUNING" <<'JSON'
import json, sys
from pathlib import Path
out = {}
for path in sys.argv[1:]:
    if Path(path).is_file():
        out.update(json.loads(Path(path).read_text())["best"])
print(json.dumps(out))
JSON
}

measure() {
  local label="$1" out="$2"; shift 2
  local metrics="${out%.json}.metrics.json"
  if done_already "$metrics"; then
    step "$label already measured; FORCE=1 to redo"
    "$PY" -c "
import json; m = json.load(open('$metrics')); c = m.get('candidate_recall', {})
print(f\"  Recall@5 {m['recall']:.4f}   candidate recall @50 {c.get('@50','-')} @100 {c.get('@100','-')}\")"
    return 0
  fi
  timed "$label" "$PY" -m legalir2 predict --evaluate "$@" --output "$out"
}

compare() {
  "$PY" - "$1" "$2" "$3" <<'CMP'
import json, sys
before, after, label = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
delta = after["recall"] - before["recall"]
print(f"\n  {label}: Recall@5 {before['recall']:.4f} -> {after['recall']:.4f}  "
      f"({'up' if delta > 0 else 'DOWN'} {delta:+.4f})")
CMP
}

# --------------------------------------------------------------------- the sweep
stage_sweep() {
  say "tune the reranking stage (~30 min, one GPU pass then pure arithmetic)"
  step "candidates / passages / aggregate / aggregate_weight / rerank_blend were all"
  step "set by hand and never measured. This is the cheapest point left on the board."
  if done_already "$RERANK_TUNING"; then
    step "already swept: $("$PY" -c "import json;print(json.load(open('$RERANK_TUNING'))['best'])")"
  else
    timed "tune rerank" "$PY" -m legalir2 tune --stage rerank --limit 400 \
        --target-depth 100 --cache-passages 4 \
        "${DENSE_ARGS[@]}" --reranker-model "$BEST_RERANKER" \
        --settings "$(settings)" --output "$RERANK_TUNING"
  fi

  say "measure the tuned settings on the full validation split"
  measure "predict validation" outputs/v3-swept.json \
      "${DENSE_ARGS[@]}" --reranker-model "$BEST_RERANKER" \
      --settings "$(settings)"
  gate "if this is flat, the hand-picked settings were already near optimal and the
     remaining gap is the model, not the plumbing - which is what distil addresses.
     If it moved, the gain is free and carries into every later measurement."
}

# ------------------------------------------------------------------ distillation
stage_distil() {
  say "teacher checkpoint: Qwen3-Reranker-4B (~8 GB download)"
  step "offline only - it writes a score cache and is never loaded at prediction time"
  if done_already models2/teacher/config.json; then
    step "already downloaded"
  else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
    timed "download teacher" "$PY" -m legalir2 download --which teacher
    timed "verify hashes" "$PY" -m legalir2 verify-models
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  say "score every training group with the teacher (~3 h, 5,950 x 32 pairs)"
  step "this is the one irreplaceable artefact of this phase - back it up when it lands"
  if done_already artifacts2/groups/teacher-r2.jsonl; then
    step "already scored"
  else
    timed "teacher scores" "$PY" -m legalir2 teacher \
        --groups artifacts2/groups/train-r2.jsonl \
        --teacher-model models2/teacher \
        --batch-size 8 --max-length 1024 \
        --output artifacts2/groups/teacher-r2.jsonl
  fi
  wc -l artifacts2/groups/teacher-r2.jsonl

  say "retrain the student with the teacher's distribution added (~2 h)"
  step "loss = listwise cross entropy + 0.4 x KL against the teacher over the same 32"
  if done_already models2/reranker-distilled/config.json; then
    step "already trained; FORCE=1 to retrain"
  else
    timed "train-ce distilled" "$PY" -m legalir2 train-ce \
        --groups artifacts2/groups/train-r2.jsonl \
        --teacher-scores artifacts2/groups/teacher-r2.jsonl \
        --distill-weight 0.4 --distill-temperature 1.0 \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --output models2/reranker-distilled
  fi

  say "measure the distilled student"
  measure "predict validation" outputs/v3-distilled.json \
      "${DENSE_ARGS[@]}" --reranker-model models2/reranker-distilled \
      --settings "$(settings)"
  [ -f outputs/v3-swept.metrics.json ] \
    && compare outputs/v3-swept.metrics.json outputs/v3-distilled.metrics.json "distillation"

  say "and the student ensembled with its undistilled sibling"
  step "distilled and undistilled disagree on different questions; both are ours already"
  measure "predict validation" outputs/v3-distilled-ens.json \
      "${DENSE_ARGS[@]}" \
      --reranker-model models2/reranker-distilled "$BEST_RERANKER" \
      --settings "$(settings)"
  compare outputs/v3-distilled.metrics.json outputs/v3-distilled-ens.metrics.json "+ ensemble"

  say "deployed parameter count - the teacher must not appear here"
  "$PY" -m legalir2 audit "${DENSE_ARGS[@]}" \
      --reranker-model models2/reranker-distilled "$BEST_RERANKER"
  gate "the audit total must be under 4,000,000,000 and must NOT include the 4B teacher.
     If a teacher path appears in that output, something is loading it at prediction
     time and the submission would be non-compliant - stop and fix it before packing."
}

case "$STAGE" in
  sweep)  stage_sweep ;;
  distil) stage_distil ;;
  all)    stage_sweep; stage_distil ;;
  *)      echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
