#!/usr/bin/env bash
# Phase eight. Three changes, in the order that keeps each one measurable.
#
#   bash gpu-phase8.sh probe     # ~1.5 h  does a SECOND reranker help? no training
#   bash gpu-phase8.sh doc       # ~3.5 h  document-loss retrained from the BASE checkpoint
#   bash gpu-phase8.sh stack     # ~1 h    refit the fusion, out of fold, on the winner
#   bash gpu-phase8.sh final     # ~2 h    retrain on ALL 7,000 labels and submit
#
# WHY THIS ORDER
# --------------
# `final` trains on the 1,050-question holdout. After it runs there is no held-out data
# left and nothing can be measured again - so everything that needs a decision has to be
# decided before it. It is last, it is gated, and it is not a place to try ideas.
#
# `probe` is first because it costs no training at all. The pipeline already combines
# several rerankers as a weighted blend of min-max normalised rankings, so the only
# question - do two different model families make different mistakes - can be answered
# with two prediction runs.
#
# THE ONE TRAP
# ------------
# `final_order` short-circuits into the stacker whenever one is active, and
# `build_features` reads passage_scores[0] only. With a stacker loaded, reranker two is
# scored, paid for, and then ignored. Phase six measured "a 2B model filtered through a
# stacker fitted on 0.6B features" and reported it as "the 2B does not help". So `probe`
# runs with NO stacker on both sides, and `stack` is a separate stage afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-probe}"
FORCE="${FORCE:-0}"
PARTNERS="${PARTNERS:-partner_qwen partner_pho partner_viranker}"
BASE_RERANKER="${BASE_RERANKER:-models2/reranker}"     # the untrained AITeamVN checkpoint
SHIPPED="${SHIPPED:-models2/reranker-g32}"
# NOT "GROUPS": bash owns that name. It is a special ARRAY holding the current user's
# group ids, so the assignment lands in element 0 and $DOC_GROUPS expands to a gid - "0"
# for root. stage_doc then tested for a file called "0", found none, printed
# "0 missing" and returned. The document-loss experiment silently never ran.
DOC_GROUPS="${DOC_GROUPS:-artifacts2/groups/train-doc.jsonl}"
DENSE=(--dense artifacts2/dense --embedding-model models2/embedding)

LOG="$ROOT/outputs/phase8.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
backup() {
  command -v rclone >/dev/null || return 0
  rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -1 || true
}

# Fusion weights only. No stacker: see THE ONE TRAP above.
BLEND() {
  "$PY" - "$@" <<'JSON'
import json, sys
from pathlib import Path
out = json.loads(Path("outputs/v2-fusion-tuning.json").read_text())["best"]
out["stacker"] = None
for pair in sys.argv[1:]:
    key, _, value = pair.partition("=")
    out[key] = json.loads(value)
print(json.dumps(out))
JSON
}

# ------------------------------------------------------------------ 1. the probe
stage_probe() {
  stage_now "probe"
  [ -d "$SHIPPED" ] || { step "$SHIPPED missing - restore from Drive first"; return 1; }

  say "baseline: the shipped reranker alone, no stacker"
  step "this is the number every partner is compared against. It is NOT 0.9493 -"
  step "that figure includes the stacker, which is deliberately off here."
  if ! done_already outputs/v8-solo.metrics.json; then
    timed "predict solo" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
        --reranker-model "$SHIPPED" --settings "$(BLEND)" --output outputs/v8-solo.json
  fi

  for which in $PARTNERS; do
    say "partner: $which"
    local dir="models2/$which"
    if [ ! -f "$dir/config.json" ]; then
      ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        timed "download $which" "$PY" -m legalir2 download --which "$which" ) || {
          step "could not fetch $which - skipping it"; continue; }
    fi

    say "  deployed parameter budget with this partner"
    "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$SHIPPED" "$dir" || {
        step "audit failed for $which - skipping"; continue; }

    # stage2_candidates = candidates, so the partner scores every shortlisted document
    # rather than only the first model's best 25. At depth 50 that costs little and it
    # is the only setting under which the partner can rescue something the shipped
    # model ranked badly - which is the entire hypothesis being tested.
    if ! done_already "outputs/v8-$which.metrics.json"; then
      timed "predict $which" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
          --reranker-model "$SHIPPED" "$dir" \
          --settings "$(BLEND stage2_candidates=50)" \
          --output "outputs/v8-$which.json" || { step "$which failed - skipping"; continue; }
    fi

    say "  $which ensembled vs the shipped reranker alone, paired"
    "$PY" -m legalir2 compare --before outputs/v8-solo.diagnostics.json \
        --after "outputs/v8-$which.diagnostics.json" --label "+ $which"
    backup
  done

  gate "adopt a partner only if its interval CLEARS ZERO. If one does, run 'stack' to
     refit the fusion with it - until then the stacker cannot see it at all and the
     0.9493 configuration is still the one to ship. If none does, the ensemble
     hypothesis is answered and 'doc' is the remaining move."
}

# ------------------------------------------ 2. document loss, from the BASE checkpoint
stage_doc() {
  stage_now "doc"
  [ -f "$DOC_GROUPS" ] || { step "$DOC_GROUPS missing - restore artifacts2/groups from Drive"; return 1; }
  [ -f "$BASE_RERANKER/config.json" ] || {
      ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
        timed "download base reranker" "$PY" -m legalir2 download --which reranker ); }

  say "why this is not a repeat of the run that measured -0.0032"
  cat <<'WHY'
  That run started from reranker-g32, which was already fine-tuned on these exact
  5,950 questions. On its own training data it ranked the gold document above all 31
  negatives, so softplus(negative - gold) underflowed and roughly nine steps in ten
  produced a loss of 0.0000. Three hours of GPU trained on perhaps 1,200 effective
  examples, while AdamW - which normalises by sqrt(v) and so steps at nearly full
  learning rate on vanishing gradients - walked the weights around on the rest.

  A cutoff-aware loss needs gold documents sitting below fifth place. Those exist in
  the base checkpoint's rankings. They had been trained out of g32's.
WHY

  local out=models2/reranker-doc-base
  if done_already "$out/config.json"; then
    step "already trained; FORCE=1 to retrain"
  else
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --reranker-model "$BASE_RERANKER" --groups "$DOC_GROUPS" --document-loss \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --max-length 768 --output "$out"
  fi

  if ! done_already outputs/v8-doc-base.metrics.json; then
    timed "predict validation" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
        --reranker-model "$out" --settings "$(BLEND)" --output outputs/v8-doc-base.json
  fi

  say "against the shipped reranker alone, paired"
  "$PY" -m legalir2 compare --before outputs/v8-solo.diagnostics.json \
      --after outputs/v8-doc-base.diagnostics.json --label "document loss, base init"
  backup
  gate "watch the multi-gold subset in particular - that is what the objective targets,
     and it moved +0.0287 even in the broken run. One standard error on 93 questions
     is 0.036, so it needs to be large there to be visible at all."
}

# ------------------------------------------------- 3. refit the fusion on the winner
stage_stack() {
  stage_now "stack"
  local model="${MODEL:-$SHIPPED}"
  say "refit the learned fusion for $model, scored OUT OF FOLD"
  step "the phase-four stacker was fitted on the shipped reranker's score distribution."
  step "a different reranker produces different numbers, and reusing those weights is"
  step "what made the 2B comparison uninterpretable."
  timed "stack" "$PY" -m legalir2 stack --limit 2000 "${DENSE[@]}" \
      --reranker-model $model --settings "$(BLEND)" \
      --cache artifacts2/groups/stack-v8.jsonl --output outputs/v8-stacker.json

  local settings
  settings="$(BLEND)"
  settings="$("$PY" -c "
import json,sys; s=json.loads(sys.argv[1]); s['stacker']='outputs/v8-stacker.json'
print(json.dumps(s))" "$settings")"
  timed "predict validation" "$PY" -m legalir2 predict --evaluate "${DENSE[@]}" \
      --reranker-model $model --settings "$settings" --output outputs/v8-stacked.json

  say "against the 0.9493 configuration, paired"
  "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/v8-stacked.diagnostics.json --label "phase 8 vs 0.9493"
  backup
  gate "this is the decision point. Ship only if the interval clears zero. Whatever
     configuration wins here is the one 'final' retrains - and after that nothing
     can be measured again."
}

# --------------------------------------------------- 4. all 7,000 labels, then submit
stage_final() {
  stage_now "final"
  [ "${I_HAVE_CHOSEN:-0}" = "1" ] || {
    cat <<'STOP'

  REFUSING TO RUN.

  This stage trains on artifacts/splits/validation.json - the 1,050 questions every
  comparison in this project has used. Afterwards there is no held-out data: the next
  measurement would be the model scoring its own training set, and it would look
  excellent no matter what.

  It is worth doing exactly once, at the end, on a configuration already chosen. About
  18% more training data, taken on principle rather than measured.

  When the configuration is settled and you will not be testing anything else:

      MODEL=<the chosen reranker> I_HAVE_CHOSEN=1 bash gpu-phase8.sh final

STOP
    return 2; }

  local model="${MODEL:?set MODEL to the reranker this should retrain}"
  say "merging both splits into artifacts/splits/all.json"
  "$PY" - <<'MERGE'
import json
from pathlib import Path
out, splits = {}, Path("artifacts/splits")
for name in ("train.json", "validation.json"):
    out.update(json.loads((splits / name).read_text(encoding="utf-8")))
(splits / "all.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"  {len(out)} questions")
MERGE

  say "re-mine over all 7,000 (~28 min)"
  if ! done_already artifacts2/groups/all-doc.jsonl; then
    timed "mine" "$PY" -m legalir2 mine --questions artifacts/splits/all.json \
        --depth 50 --negatives 31 --positives 3 \
        --settings "$(BLEND)" --output artifacts2/groups/all-doc.jsonl
  fi

  say "retrain on everything (~3.5 h)"
  local out=models2/reranker-final
  if ! done_already "$out/config.json"; then
    # DOC_LOSS=0 reruns the recipe that produced reranker-g32 - the listwise objective,
    # initialised from the base checkpoint - on 18% more data. That is the right choice
    # when the document loss has not beaten it: retraining with an objective that just
    # failed its test, on data you can no longer measure against, is how a final
    # submission ends up worse than the one it replaces.
    local objective=()
    [ "${DOC_LOSS:-1}" = "1" ] && objective=(--document-loss)
    step "objective: $([ "${DOC_LOSS:-1}" = 1 ] && echo 'document-level Recall@5' || echo 'listwise softmax (the g32 recipe)')"
    step "initialised from: $model"
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --reranker-model "$model" --groups artifacts2/groups/all-doc.jsonl \
        "${objective[@]}" \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --max-length 768 --output "$out"
  fi

  say "refit the fusion on all 7,000, reported out of fold"
  timed "stack" "$PY" -m legalir2 stack --limit 7000 "${DENSE[@]}" \
      --reranker-model "$out" --settings "$(BLEND)" \
      --cache artifacts2/groups/stack-final.jsonl --output outputs/final-stacker.json

  say "deployed parameter budget - this is a competition rule, not a guideline"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model "$out"

  local settings
  settings="$("$PY" -c "
import json,sys; s=json.loads(sys.argv[1]); s['stacker']='outputs/final-stacker.json'
print(json.dumps(s))" "$(BLEND)")"
  say "public test"
  timed "predict public" "$PY" -m legalir2 predict --questions public-official.json \
      "${DENSE[@]}" --reranker-model "$out" --settings "$settings" \
      --output outputs/submission-final.json
  "$PY" -m legalir2 validate --predictions outputs/submission-final.json
  "$PY" -m legalir2 pack --predictions outputs/submission-final.json \
      --output outputs/submission-final.zip
  ls -lh outputs/submission-final.zip
  backup
  gate "there is no validation number for this run and there cannot be. Judge it on the
     leaderboard, and keep the 0.9493 zip - if this scores lower, that is the one that
     still stands."
}

# --------------------------------------------------------------- 5. unattended run
# Exit 0 when `after` beats `before` on a paired bootstrap interval that excludes zero.
# The same rule every gate in this project states in prose - here it is machine readable,
# so an overnight run adopts a change for the same reason a person would and not because
# one mean happened to be larger.
decide() {
  "$PY" - "$1" "$2" <<'PY'
import json, sys
from legalir2.compare import bootstrap, per_question
before = per_question(json.load(open(sys.argv[1], encoding="utf-8")))
after = per_question(json.load(open(sys.argv[2], encoding="utf-8")))
keys = sorted(set(before) & set(after))
if not keys:
    print("    no overlapping questions"); raise SystemExit(1)
d = [after[k] - before[k] for k in keys]
low, high = bootstrap(d, resamples=4000)
mean = sum(d) / len(d)
print(f"    {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  "
      f"{'ADOPT' if low > 0 else 'keep incumbent'}")
raise SystemExit(0 if low > 0 else 1)
PY
}

stage_auto() {
  stage_now "auto"
  say "unattended: probe, document loss, then a submission from whatever wins"
  step "nothing that trains on the holdout runs here. 'final' stays manual."

  stage_probe || true
  stage_doc || true

  say "choosing a configuration"
  local model="$SHIPPED" partner="" best=""
  for which in $PARTNERS; do
    [ -f "outputs/v8-$which.diagnostics.json" ] || continue
    printf '  %s\n' "$which"
    if decide outputs/v8-solo.diagnostics.json "outputs/v8-$which.diagnostics.json"; then
      best="$which"
    fi
  done
  [ -n "$best" ] && partner="models2/$best"

  if [ -f outputs/v8-doc-base.diagnostics.json ]; then
    printf '  document loss, base init\n'
    decide outputs/v8-solo.diagnostics.json outputs/v8-doc-base.diagnostics.json \
      && model=models2/reranker-doc-base
  fi

  local chosen="$model${partner:+ $partner}"
  step "chosen rerankers: $chosen"
  if [ "$chosen" = "$SHIPPED" ]; then
    step "nothing beat the incumbent, so this rebuilds the 0.9493 configuration."
    step "that is a real answer, not a failure: two more hypotheses are now closed."
  fi

  local settings
  if [ -n "$partner" ]; then
    # The stacker and a second reranker are mutually exclusive as the code stands.
    # build_features reads passage_scores[0] only, so loading a stacker discards the
    # partner and reproduces the single-model ranking exactly - 1,050 of 1,050
    # predictions bit-identical, which is what the first run of this stage printed and
    # what made its submission a copy of the one already on the leaderboard.
    say "a partner was adopted, so the stacker stays OFF"
    step "with a stacker loaded the partner is scored, paid for, and then ignored."
    step "until build_features consumes both rerankers, these two are exclusive."
    settings="$(BLEND stage2_candidates=50)"
    say "the ensemble against the SHIPPED 0.9493 configuration, paired"
    step "the probe compared it against the no-stacker baseline. This is the comparison"
    step "that decides a submission: ensemble-without-stacker vs single-with-stacker."
    "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
        --after "outputs/v8-$best.diagnostics.json" --label "ensemble vs 0.9493" || true
  else
    say "refit the fusion for this configuration, out of fold"
    MODEL="$chosen" stage_stack || true
    settings="$("$PY" -c "
import json,sys; s=json.loads(sys.argv[1]); s['stacker']='outputs/v8-stacker.json'
print(json.dumps(s))" "$(BLEND)")"
  fi

  say "public test with the chosen configuration"
  "$PY" -m legalir2 audit "${DENSE[@]}" --reranker-model $chosen
  timed "predict public" "$PY" -m legalir2 predict --questions public-official.json \
      "${DENSE[@]}" --reranker-model $chosen --settings "$settings" \
      --output outputs/submission-v8.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v8.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v8.json \
      --output outputs/submission-v8.zip
  ls -lh outputs/submission-v8.zip
  command -v rclone >/dev/null && rclone copy outputs/submission-v8.zip \
      "$RESULTS/outputs" 2>&1 | tail -1
  backup

  gate "the zip is built from the best configuration the paired tests supported. If the
     'phase 8 vs 0.9493' interval printed by 'stack' spans zero, this submission is not
     better than the one you already have - upload the old one and keep the slot."
}

# ----------------------------------------------- 6. doc, then final, decision included
stage_all() {
  stage_now "all"
  say "document loss, then the final retrain on all 7,000 labels"
  step "roughly 9.5 h. The objective 'final' uses is decided by the paired test below,"
  step "not chosen in advance."
  stage_doc || true

  local doc_loss=0
  if [ -f outputs/v8-doc-base.diagnostics.json ]; then
    say "did the document loss beat the shipped reranker?"
    if decide outputs/v8-solo.diagnostics.json outputs/v8-doc-base.diagnostics.json; then
      doc_loss=1
      step "adopted - 'final' trains the document objective on all 7,000"
    else
      step "not adopted - 'final' re-runs the listwise recipe that produced g32,"
      step "on 18% more data. Retraining with an objective that just failed its test,"
      step "against data you can no longer measure, is how a final submission ends up"
      step "worse than the one it replaces."
    fi
  else
    step "no document-loss result, so 'final' uses the listwise recipe"
  fi

  # Always initialised from the BASE checkpoint, whichever objective won. Continuing
  # from reranker-doc-base would train it a second time on the 5,950 questions it has
  # already seen, which is the saturation that made the first document-loss run
  # produce a loss of 0.0000 for nine steps in ten.
  MODEL="$BASE_RERANKER" DOC_LOSS="$doc_loss" stage_final
}

case "$STAGE" in
  all)   stage_all ;;
  auto)  stage_auto ;;
  probe) stage_probe ;;
  doc)   stage_doc ;;
  stack) stage_stack ;;
  final) stage_final ;;
  *)     echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
