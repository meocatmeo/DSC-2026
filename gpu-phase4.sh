#!/usr/bin/env bash
# Phase four: 0.9433 -> as high as this system goes.
#
#   bash gpu-phase4.sh multipos   # ~2.5 h  multi-positive loss, the biggest known pool
#   bash gpu-phase4.sh stack      # ~1.0 h  learned rank fusion (mostly CPU)
#   bash gpu-phase4.sh sweep      # ~1.5 h  rerank knobs incl. context_radius
#   bash gpu-phase4.sh ceiling    # ~4.0 h  tuned embedder + RE-MINED negatives
#   bash gpu-phase4.sh            # all four, in order
#
# Ordered by evidence, not by ambition. The first three are measured wins waiting to be
# collected; `ceiling` is the speculative one and goes last so a failure there cannot
# cost the rest.
#
# WHY `ceiling` LOOKS LIKE A CONTRADICTION AND IS NOT
# ---------------------------------------------------
# Phase two measured three separate regressions from improving retrieval: the tuned
# bi-encoder reached candidate recall@100 of 0.9903 - the best of the project - and the
# final score FELL 0.9371 -> 0.9333. Deeper candidates fell further, to 0.9276.
#
# The cause is the one this project has already paid for once: the reranker was trained
# on negatives mined through the BASE retrieval. Point it at a different candidate
# distribution and it is being asked to reject documents it has never seen. Fixing that
# exact mismatch is what took 0.9223 -> 0.9433. So `ceiling` does not just switch on the
# better index - it re-mines the negatives THROUGH that index and retrains. Anything less
# repeats the measured regression.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
export HF_HUB_DISABLE_TELEMETRY=1
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
LOG="$ROOT/outputs/phase4.log"
command -v rclone >/dev/null || { echo "rclone is required; configure it before starting"; exit 1; }
rclone lsd "$RESULTS" >/dev/null 2>&1 || {
  echo "cannot access $RESULTS; check ~/.config/rclone/rclone.conf"; exit 1;
}
source "$ROOT/logship.sh"
stage_now "$STAGE"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

valid_groups() {
  "$PY" - "$1" <<'CHECK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); stats = Path(str(p).replace('.jsonl', '.stats.json'))
if not p.is_file() or not stats.is_file(): raise SystemExit(1)
s = json.loads(stats.read_text())
if s.get('positives') != 3 or s.get('negatives') != 31 or s.get('depth') != 50: raise SystemExit(1)
with p.open(encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if not row.get('positives_by_gold'): raise SystemExit(1)
CHECK
}

valid_model() {
  "$PY" - "$1" <<'CHECK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'training.json'
if not p.is_file(): raise SystemExit(1)
r = json.loads(p.read_text())
if not r.get('document_loss') or r.get('head') != 'sequence' or r.get('max_length') != 768: raise SystemExit(1)
if Path(str(r.get('base', ''))).name != 'reranker-g32': raise SystemExit(1)
CHECK
}

BASE_FUSION="outputs/v2-fusion-tuning.json"
RERANK_TUNING="outputs/v7-rerank-tuning-doc.json"
STACKER="outputs/v7-stacker-doc.json"

# Phase two's winner, and the index its negatives were mined through. These travel
# together: a reranker and the candidate distribution it was trained against.
BEST="models2/reranker-g32"
DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)

settings() {   # fusion weights + swept knobs + stacker, merged. One source of truth.
  "$PY" - "$@" <<'JSON'
import json, sys
from pathlib import Path
out = {}
for path in sys.argv[1:]:
    p = Path(path)
    if not p.is_file():
        continue
    record = json.loads(p.read_text())
    if "stacker" in p.stem:
        out["stacker"] = str(p)
    else:
        out.update(record["best"])
print(json.dumps(out))
JSON
}
# Which reranker later stages build on. Chosen by MEASURED score from the recorded run
# manifests, never by "this directory exists" - reranker-mp exists and is very slightly
# worse, so existence would silently propagate the loser through every later stage.
best_reranker() {
  "$PY" - <<'PICK'
import glob, json
from pathlib import Path
best, score = "models2/reranker-g32", -1.0
for metrics_path in glob.glob("outputs/v[247]-*.metrics.json"):
    run_path = Path(metrics_path.replace(".metrics.json", ".run.json"))
    if not run_path.is_file():
        continue
    try:
        recall = json.loads(Path(metrics_path).read_text())["recall"]
        models = (json.loads(run_path.read_text()).get("models") or {}).get("rerankers") or []
    except Exception:
        continue
    if len(models) == 1 and recall > score and Path(models[0]).is_dir():
        best, score = models[0], recall
print(best)
PICK
}

base_settings()  { settings "$BASE_FUSION" "$RERANK_TUNING"; }
full_settings()  { settings "$BASE_FUSION" "$RERANK_TUNING" "$STACKER"; }

measure() {
  local label="$1" out="$2"; shift 2
  local metrics="${out%.json}.metrics.json"
  local expected_model="${MEASURE_MODEL:-}"
  local expected_questions="${MEASURE_QUESTIONS:-}"
  local valid=0
  if done_already "$metrics" && [ -n "$expected_model" ]; then
    "$PY" - "$out" "$expected_model" "$expected_questions" <<'CHECK' && valid=1 || true
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1] + '.run.json').read_text())
models = r.get('models', {}).get('rerankers', [])
ok_model = models and Path(models[0]).name == Path(sys.argv[2]).name
ok_questions = not sys.argv[3] or Path(str(r.get('questions', ''))).name == Path(sys.argv[3]).name
raise SystemExit(0 if ok_model and ok_questions else 1)
CHECK
  elif done_already "$metrics"; then
    valid=1
  fi
  if [ "$valid" = 1 ]; then
    step "$label already measured; FORCE=1 to redo"
    "$PY" -c "
import json; m = json.load(open('$metrics')); c = m.get('candidate_recall', {})
print(f\"  Recall@5 {m['recall']:.4f}   candidate recall @50 {c.get('@50','-')} @100 {c.get('@100','-')}\")"
    return 0
  fi
  timed "$label" "$PY" -m legalir2 predict --evaluate "$@" --output "$out"
}

# Paired, not a difference of means. Both systems score the same 1050 questions, so
# question difficulty cancels and only the disagreements carry information. Comparing
# means has a standard error of ~0.0075 here, which makes every difference this project
# is now chasing invisible - and invites reading noise as a result, which has already
# happened once.
compare() {
  local before="${1%.metrics.json}.diagnostics.json"
  local after="${2%.metrics.json}.diagnostics.json"
  if [ -f "$before" ] && [ -f "$after" ]; then
    "$PY" -m legalir2 compare --before "$before" --after "$after" --label "$3"
  else
    "$PY" - "$1" "$2" "$3" <<'CMP'
import json, sys
before, after, label = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
delta = after["recall"] - before["recall"]
print(f"\n  {label}: Recall@5 {before['recall']:.4f} -> {after['recall']:.4f}  "
      f"({'up' if delta > 0 else 'DOWN'} {delta:+.4f})  (means only - no diagnostics)")
CMP
  fi
}

# ------------------------------------------------------- 1. multi-positive loss
stage_multipos() {
  say "re-mine and train document-level Recall@5 loss on the shipped 0.6B reranker"
  step "the old groups pool every gold document's passages into one flat list, which"
  step "cannot express 'these three are all correct' - only 'sample one of them'"
  if done_already artifacts2/groups/train-doc.jsonl && valid_groups artifacts2/groups/train-doc.jsonl; then
    step "already mined (validated positives=3, negatives=31, depth=50)"
  else
    step "missing or stale training groups; rebuilding"
    timed "mine" "$PY" -m legalir2 mine --depth 50 --negatives 31 --positives 3 \
        --settings "$(base_settings)" --output artifacts2/groups/train-doc.jsonl
  fi
  "$PY" -c "
import json; s = json.load(open('artifacts2/groups/train-doc.stats.json'))
print(f\"  {s['questions']} groups, {s.get('multi_gold', 0)} of them multi-gold\")
assert 'multi_gold' in s, 'stale miner: re-pull the code, these groups lack positives_by_gold'"

  say "retrain with the target mass split across every gold document (~2 h)"
  step "8.9% of questions, 34% of the lost recall. Single-gold groups are bit-identical"
  step "to the old objective, so this cannot regress the other 91%."
  if done_already models2/reranker-doc/config.json && valid_model models2/reranker-doc; then
    step "already trained (validated 0.6B document-loss checkpoint)"
  else
    step "missing or stale document checkpoint; retraining from reranker-g32"
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --reranker-model models2/reranker-g32 \
        --groups artifacts2/groups/train-doc.jsonl --document-loss \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --max-length 768 --output models2/reranker-doc
  fi

  MEASURE_MODEL=models2/reranker-doc MEASURE_QUESTIONS=validation.json \
  measure "predict validation" outputs/v7-document.json \
      "${DENSE[@]}" --reranker-model models2/reranker-doc --settings "$(base_settings)"
  [ -f outputs/v2-g32.metrics.json ] \
    && compare outputs/v2-g32.metrics.json outputs/v7-document.metrics.json "document-level loss"
  "$PY" - <<'SPLIT'
import json
from pathlib import Path
p = Path("outputs/v7-document.diagnostics.json")
if not p.is_file():
    raise SystemExit("  (no diagnostics yet)")
d = json.loads(p.read_text())
single = [r for r in d.values() if len(r["relevant"]) == 1]
multi  = [r for r in d.values() if len(r["relevant"]) > 1]
def score(rows):
    return sum(len(set(r["relevant"]) & set(r["predicted"][:5])) / len(r["relevant"])
               for r in rows) / max(1, len(rows))
print(f"\n  single-gold {score(single):.4f}  (was 0.9551)")
print(f"  multi-gold  {score(multi):.4f}  (was 0.7527)  <- the number this stage targets")
SPLIT
  gate "read the PAIRED comparison above, not the means. With 93 multi-gold questions,
     one standard error on that subset is 0.036 - a change smaller than about 0.07 there
     is not measurable at all, and the full-set standard error is 0.0075. If the 95%
     interval spans zero, this stage neither helped nor hurt and the right move is to
     keep whichever checkpoint you already have and go on to 'stack'."
}

# --------------------------------------------------------- 2. learned rank fusion
stage_stack() {
  say "fit the learned rank fusion (~45 min GPU to collect, seconds to fit)"
  step "replaces 'reranker x 1.0 + everything-else x 0.02', a ratio never fitted"
  local model="models2/reranker-doc"
  [ -d "$model" ] || model="$(best_reranker)"
  step "using $model (highest measured validation score, not most recent)"
  if done_already "$STACKER"; then
    step "already fitted"
  else
    timed "stack" "$PY" -m legalir2 stack --limit 2000 \
        "${DENSE[@]}" --reranker-model "$model" \
        --settings "$(base_settings)" \
        --cache artifacts2/groups/stack-doc.jsonl --output "$STACKER"
  fi

  MEASURE_MODEL="$model" MEASURE_QUESTIONS=validation.json \
  measure "predict validation" outputs/v7-stacked.json \
      "${DENSE[@]}" --reranker-model "$model" --settings "$(full_settings)"
  compare outputs/v7-document.metrics.json outputs/v7-stacked.metrics.json "learned fusion" \
    2>/dev/null || compare outputs/v2-g32.metrics.json outputs/v7-stacked.metrics.json "learned fusion"
  [ -f outputs/v4-stacked.metrics.json ] && \
    compare outputs/v4-stacked.metrics.json outputs/v7-stacked.metrics.json "new 0.6B system vs shipped 0.9493 baseline"
  gate "the printed weights are the diagnosis: if the retrieval features carry real
     weight, the 0.02 constant was throwing away signal, exactly as predicted. If they
     come out near zero, the reranker really was sufficient and this stage is a no-op."
}

# ------------------------------------------------------------------- 3. the sweep
stage_sweep() {
  say "sweep the reranking knobs, including context_radius (~1.5 h)"
  step "context_radius=0 means the reranker judges a 320-word chunk in isolation while"
  step "the model accepts 2304 tokens. Each radius needs its own scoring pass."
  local model="models2/reranker-doc"
  [ -d "$model" ] || model="$(best_reranker)"
  step "using $model"
  if done_already "$RERANK_TUNING"; then
    step "already swept"
  else
    timed "tune rerank" "$PY" -m legalir2 tune --stage rerank --limit 400 \
        --target-depth 100 --cache-passages 4 --context-radius 0 1 \
        "${DENSE[@]}" --reranker-model "$model" \
        --settings "$(base_settings)" --output "$RERANK_TUNING"
  fi
  MEASURE_MODEL="$model" MEASURE_QUESTIONS=validation.json \
  measure "predict validation" outputs/v7-swept.json \
      "${DENSE[@]}" --reranker-model "$model" --settings "$(full_settings)"
  gate "do NOT expect candidates=100 to win: v2-deep already measured that as a
     regression (0.9371 -> 0.9276). If the sweep picks it anyway, trust the sweep - it
     is measuring the current reranker, v2-deep measured the old one."
}

# ------------------------------------------------- 4. the ceiling, done properly
stage_ceiling() {
  say "the speculative stage: raise the ceiling AND re-mine through it"
  step "tuned embedder reached candidate@100 0.9903 but scored 0.9333, because the"
  step "reranker had never seen its candidates. Re-mining is what makes it usable."
  if [ ! -f models2/embedding-tuned/config.json ] || [ ! -f artifacts2/dense-tuned/manifest.json ]; then
    step "no tuned embedder or index on this box - restore them or run gpu-phase2.sh biencoder"
    return 0
  fi
  local TUNED=(--dense artifacts2/dense-tuned --embedding-model models2/embedding-tuned)
  local TF="outputs/v7-fusion-tuned-embedder.json"

  if done_already "$TF"; then step "fusion already tuned for the tuned index"; else
    timed "tune fusion" "$PY" -m legalir2 tune --stage fusion --limit 400 \
        --target-depth 100 "${TUNED[@]}" --output "$TF"
  fi

  say "re-mine negatives THROUGH the tuned index - this is the whole point of the stage"
  if done_already artifacts2/groups/train-tuned.jsonl; then step "already mined"; else
    timed "mine" "$PY" -m legalir2 mine --negatives 31 "${TUNED[@]}" \
        --settings "$(settings "$TF" "$RERANK_TUNING")" \
        --output artifacts2/groups/train-tuned.jsonl
  fi

  say "retrain against that distribution (~2 h)"
  if done_already models2/reranker-tuned-idx/config.json; then step "already trained"; else
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --groups artifacts2/groups/train-tuned.jsonl --max-positives 4 \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --output models2/reranker-tuned-idx
  fi

  say "re-fit the stacker for this pairing, then measure"
  local ST2="outputs/v7-stacker-tuned.json"
  if done_already "$ST2"; then step "already fitted"; else
    timed "stack" "$PY" -m legalir2 stack --limit 2000 "${TUNED[@]}" \
        --reranker-model models2/reranker-tuned-idx \
        --settings "$(settings "$TF" "$RERANK_TUNING")" \
        --cache artifacts2/groups/stack-tuned.jsonl --output "$ST2"
  fi
  MEASURE_MODEL=models2/reranker-tuned-idx MEASURE_QUESTIONS=validation.json \
  measure "predict validation" outputs/v7-ceiling.json \
      "${TUNED[@]}" --reranker-model models2/reranker-tuned-idx \
      --settings "$(settings "$TF" "$RERANK_TUNING" "$ST2")"
  compare outputs/v7-swept.metrics.json outputs/v7-ceiling.metrics.json "tuned index, re-mined" \
    2>/dev/null || true
  gate "if this still loses to the base index after re-mining, the tuned embedder is
     genuinely worse for this task despite its higher candidate recall - which would be
     a real finding, and the answer is to keep the base index and stop here."
}

audit_budget() {
  say "deployed parameter budget"
  step "the stacker is arithmetic over scores - it adds nothing to this count"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$(best_reranker)"
}

case "$STAGE" in
  multipos) stage_multipos ;;
  stack)    stage_stack ;;
  sweep)    stage_sweep ;;
  ceiling)  stage_ceiling ;;
  audit)    audit_budget ;;
  all)      stage_multipos; stage_stack; stage_sweep; stage_ceiling; audit_budget ;;
  *)        echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
