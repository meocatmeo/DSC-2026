#!/usr/bin/env bash
# Fit the stacker on reranker scores it has not memorised.
#
#   bash gpu-calibrate.sh            # train, collect, fit, measure  (~3.8 h)
#   bash gpu-calibrate.sh train      # ~1.6 h  a reranker on HALF the training questions
#   bash gpu-calibrate.sh collect    # ~1.4 h  score the OTHER half with it
#   bash gpu-calibrate.sh fit        # ~1 min  CPU
#   bash gpu-calibrate.sh measure    # ~45 min paired against the 0.9508 submission
#   bash gpu-calibrate.sh submit     # ~40 min
#
# THE MECHANISM
# -------------
# v9-stacker-ens.json was fitted on 2,000 training questions - the same questions
# reranker-g32 was fine-tuned on. On its own training data a reranker separates gold
# from negatives far more cleanly than on anything unseen, so the fit over-trusts it:
#
#   rr_max     +1.5556        the reranker
#   fused_rr   +0.2874        the fusion's own ranking
#
# At inference those weights are wrong in a specific direction, and the loss map shows
# the consequence: 14 gold documents were inside the FUSION's top five and the ranking
# stage pushed them out. 0.0089 of recall, actively destroyed, 16% of all loss.
#
# So: train a reranker on fold A, score fold B with it, and fit the stacker on those
# out-of-fold features. The weights then describe how a reranker behaves on questions
# it has never seen - which is the only situation that ever occurs at inference.
#
# The partner model needs none of this. Qwen3-Reranker-0.6B is off the shelf and has
# memorised nothing, so its features are already honest.
#
# WHAT WOULD MAKE THIS WRONG
# --------------------------
# Phase five measured a cross-fitted stacker at -0.0014. That comparison was itself
# broken - `cv --fit-stacker` fitted and evaluated on the same cached questions, and
# folds were keyed on question id while 16 duplicate question texts straddled the
# split. Both are fixed. But it is evidence against, and it is why this is a 3.8 hour
# experiment rather than a plan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
PARTNER="${PARTNER:-models2/partner_qwen}"
SHIPPED="${SHIPPED:-models2/reranker-g32}"
HALF="${HALF:-models2/reranker-foldA}"
BASE="${BASE:-models2/reranker}"
# NOT "GROUPS". Bash owns that name - it is a special ARRAY of the current user's
# group ids, so the assignment lands in element 0 and $DOC_GROUPS expands to a gid,
# "0" for root. The same mistake in gpu-phase8.sh silently skipped a whole stage.
DOC_GROUPS="${DOC_GROUPS:-artifacts2/groups/train-doc.jsonl}"
STACKER="${STACKER:-outputs/v10-stacker-oof.json}"
LIMIT="${LIMIT:-2000}"
DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)

LOG="$ROOT/outputs/calibrate.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
backup() { command -v rclone >/dev/null && rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -1 || true; }

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

# Fold A trains the reranker; fold B's questions are then scored by a model that has
# never seen them. Folds are keyed on normalised question TEXT, because the labelled
# set contains 16 duplicate question groups and keying on id puts copies of the same
# question on both sides - which is the leak cross-fitting exists to prevent.
split_folds() {
  "$PY" - "$DOC_GROUPS" <<'SPLIT'
import json, sys
from pathlib import Path
from legalir2.cache import folds
source = Path(sys.argv[1])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
texts = {r["id"]: r["question"] for r in rows}
assignment = folds([r["id"] for r in rows], 2, texts=texts)
a = [r for r in rows if assignment[r["id"]] == 0]
b = {r["id"] for r in rows if assignment[r["id"]] == 1}
Path("artifacts2/groups/foldA.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in a), encoding="utf-8")
Path("artifacts2/groups/foldB.ids.json").write_text(json.dumps(sorted(b)), encoding="utf-8")
print(f"  fold A: {len(a)} groups for training")
print(f"  fold B: {len(b)} questions the fold-A model will never have seen")
SPLIT
}

stage_train() {
  stage_now "train"
  [ -f "$DOC_GROUPS" ] || { step "$DOC_GROUPS missing - rclone copy $RESULTS/groups artifacts2/groups"; return 1; }
  [ -f "$BASE/config.json" ] || ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      "$PY" -m legalir2 download --which reranker )
  say "split the training questions in two"
  split_folds
  say "train a reranker on fold A only (~1.6 h)"
  step "same recipe as reranker-g32, half the data. This model exists only to produce"
  step "honest scores for fold B - it is never deployed."
  if done_already "$HALF/config.json"; then
    step "already trained; FORCE=1 to redo"
  else
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --reranker-model "$BASE" --groups artifacts2/groups/foldA.jsonl \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --max-length 768 --output "$HALF"
  fi
  backup
}

stage_collect() {
  stage_now "collect"
  [ -f "$HALF/config.json" ] || { step "train first"; return 1; }
  say "score fold B with the fold-A model, plus the partner (~1.4 h)"
  step "reranker 1 has never seen these questions, so its scores are what the stacker"
  step "will actually meet at inference. The partner is off the shelf either way."
  "$PY" - <<'PICK'
import json
from pathlib import Path
from legalir2.predict import load_questions
from legalir2.store import write_json
keep = set(json.loads(Path("artifacts2/groups/foldB.ids.json").read_text()))
records = load_questions("artifacts/splits/train.json", labeled=True)
subset = {q: v for q, v in records.items() if q in keep}
write_json("artifacts2/groups/foldB.questions.json", subset)
print(f"  {len(subset)} fold-B questions written")
PICK
  if done_already artifacts2/groups/stack-oof.jsonl; then
    step "already collected; FORCE=1 to redo"
  else
    timed "stack collect" "$PY" -m legalir2 stack --limit "$LIMIT" \
        "${DENSE[@]}" --reranker-model "$HALF" "$PARTNER" \
        --questions artifacts2/groups/foldB.questions.json --settings "$(SET)" \
        --cache artifacts2/groups/stack-oof.jsonl --output "$STACKER"
  fi
  backup
  gate "compare the printed weights against v9's. If rr_max has fallen and fused_rr,
     den_rr or lex_rr have risen, the memorisation effect was real and this stacker is
     better calibrated for unseen questions. If they are nearly identical, the effect
     was not there and the measurement below will confirm it costs nothing to check."
}

stage_fit() {
  stage_now "fit"
  timed "stack fit" "$PY" -m legalir2 stack --limit "$LIMIT" \
      "${DENSE[@]}" --reranker-model "$HALF" "$PARTNER" \
      --questions artifacts2/groups/foldB.questions.json --settings "$(SET)" \
      --cache artifacts2/groups/stack-oof.jsonl --output "$STACKER"
  say "side by side with the stacker that scored 0.9508"
  "$PY" - <<'DIFF'
import json
from pathlib import Path
new = json.loads(Path("outputs/v10-stacker-oof.json").read_text())
old = Path("outputs/v9-stacker-ens.json")
if not old.is_file():
    raise SystemExit("  (v9-stacker-ens.json not present for comparison)")
old = json.loads(old.read_text())
names = new["features"]
a = dict(zip(old["features"], old["weights"]))
b = dict(zip(names, new["weights"]))
print(f"  {'feature':<16} {'v9 (in sample)':>15} {'v10 (out of fold)':>18} {'change':>9}")
print("  " + "-" * 62)
for name in sorted(names, key=lambda n: -abs(b[n])):
    before, after = a.get(name, 0.0), b[name]
    print(f"  {name:<16} {before:>15.4f} {after:>18.4f} {after-before:>+9.4f}")
DIFF
}

stage_measure() {
  stage_now "measure"
  [ -f "$STACKER" ] || { step "fit first"; return 1; }
  say "validation: the SHIPPED rerankers under the out-of-fold stacker (~45 min)"
  step "reranker-g32 is deployed, not the fold-A model. Only the stacker's weights"
  step "come from out-of-fold data - the pipeline itself is unchanged."
  if ! done_already outputs/v10-oof.metrics.json; then
    timed "predict validation" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
        --reranker-model "$SHIPPED" "$PARTNER" --settings "$(SET "$STACKER")" \
        --output outputs/v10-oof.json
  fi
  say "against v9, the configuration that scored 0.9508"
  "$PY" -m legalir2 compare --before outputs/v9-ensemble.diagnostics.json \
      --after outputs/v10-oof.diagnostics.json --label "out-of-fold stacker vs v9"
  say "and against v4, the 0.9493 configuration"
  "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/v10-oof.diagnostics.json --label "out-of-fold stacker vs v4"
  backup
  gate "v9 scored 0.9508 public from +0.0010 on validation, and v4 scored 0.9493 from
     0.9454 - public has tracked validation twice, at about +0.004. Eighth place is
     0.9517 and seventh is 0.9535, so a validation gain of even +0.002 is worth a
     submission here. Submit on a positive mean; do not claim it is proven."
}

stage_submit() {
  stage_now "submit"
  [ -f outputs/v10-oof.metrics.json ] || { step "measure first"; return 1; }
  say "public test (~40 min)"
  timed "predict public" "$PY" -m legalir2 predict --questions public-official.json \
      "${DENSE[@]}" --reranker-model "$SHIPPED" "$PARTNER" \
      --settings "$(SET "$STACKER")" --output outputs/submission-v10.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v10.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v10.json \
      --output outputs/submission-v10.zip
  ls -lh outputs/submission-v10.zip
  command -v rclone >/dev/null && rclone copy outputs/submission-v10.zip "$RESULTS/outputs" 2>&1 | tail -1
  backup
}

case "$STAGE" in
  train)   stage_train ;;
  collect) stage_collect ;;
  fit)     stage_fit ;;
  measure) stage_measure ;;
  submit)  stage_submit ;;
  all)     stage_train; stage_collect; stage_measure ;;
  *)       echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
