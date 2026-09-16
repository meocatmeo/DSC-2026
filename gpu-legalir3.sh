#!/usr/bin/env bash
# legalir3, stage one: fix the retrieval half.
#
#   bash gpu-legalir3.sh quick      # ~40 min  tuned embedder + stacker - NO training
#   bash gpu-legalir3.sh refit      # ~45 min  refit the stacker on the tuned pipeline
#   bash gpu-legalir3.sh baseline   # ~5 min   candidate recall as it stands today
#   bash gpu-legalir3.sh mine       # ~30 min  negatives from the DENSE channel
#   bash gpu-legalir3.sh train      # ~50 min  fine-tune the bi-encoder on them
#   bash gpu-legalir3.sh index      # ~1.2 h   rebuild the dense index
#   bash gpu-legalir3.sh measure    # ~5 min   paired candidate recall vs baseline
#   bash gpu-legalir3.sh endtoend   # ~20 min  only after measure says something moved
#   bash gpu-legalir3.sh            # baseline through measure
#
# THE ARGUMENT
# ------------
# Seven interventions after 0.9493 all re-scored the same fifty documents, and the two
# that worked at all converged on the same 0.008 and the same ~20 questions. Meanwhile
# the dense channel ships AITeamVN/Vietnamese_Embedding_v2 straight from the publisher,
# and the one time a bi-encoder was trained here it was trained on negatives tiered by
# the FUSED ranking - documents the fusion found hard. At inference the bi-encoder
# produces the dense ranking over the whole corpus by itself; fusion happens afterwards.
#
# That is the same mismatch that was worth +0.0210 when it was found in the reranker,
# sitting unfixed in the other model. `mine --channel dense` is the fix.
#
# HOW TO JUDGE IT
# ---------------
# On CANDIDATE RECALL, not on final Recall@5. Reranking converts about 58% of the
# headroom it is handed - stable across all seven interventions - so a real 0.010
# retrieval gain arrives as 0.006 at the end, under the 0.0094 detection floor. Judged
# end to end, a genuine improvement here reads as noise and gets thrown away. That is
# very likely part of why this project has looked stuck.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs artifacts2/groups
export PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
EMBEDDER="${EMBEDDER:-models2/embedding}"           # the published AITeamVN weights
TUNED="${TUNED:-models2/embedding-l3}"
INDEX="${INDEX:-artifacts2/dense-l3}"
BI_GROUPS="${BI_GROUPS:-artifacts2/groups/bi-dense.jsonl}"

LOG="$ROOT/outputs/legalir3.log"
source "$ROOT/logship.sh"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

RESULTS="${RESULTS:-gdrive:DSC-2026-results}"
backup() { command -v rclone >/dev/null && rclone copy outputs "$RESULTS/outputs" 2>&1 | tail -1 || true; }

BLEND() {
  "$PY" - <<'JSON'
import json
from pathlib import Path
out = json.loads(Path("outputs/v2-fusion-tuning.json").read_text())["best"]
out["stacker"] = None
print(json.dumps(out))
JSON
}

DENSE_NOW=(--dense artifacts2/dense --embedding-model "$EMBEDDER")
DENSE_NEW=(--dense "$INDEX" --embedding-model "$TUNED")

# ------------------------------------------------- 0. the cell nobody measured
# Phase two and phase four both built a tuned embedder and a dense index to match, and
# both measured it WITHOUT a stacker:
#
#   run           emb    stk  cand@5   @50     final   conversion
#   v4-stacked    base   yes  0.8914  0.9805  0.9454     61%
#   v4-ceiling    TUNED  no   0.9133  0.9856  0.9349     30%
#   v2-bi-tuned   TUNED  no   0.9171  0.9833  0.9333     25%
#
# The tuned embedder is a large retrieval win - +0.0257 of candidate recall@5, three
# times anything the reranker work ever produced - and it lost end to end all three
# times, because the reranker's conversion halved. A reranker that is right most of the
# time improves a weak ordering and damages a strong one: it demotes gold the better
# fusion had already placed.
#
# The stacker is the component that arbitrates exactly that. It reads fused_rr, lex_rr
# and den_rr beside the reranker's scores, so it can learn when to trust the retrieval
# order over the model. On the base pipeline it is worth +0.008; on a pipeline whose
# retrieval features are much stronger it should be worth more.
#
# outputs/v4-stacker-tuned.json was fitted and then never loaded by the ceiling stage.
# This runs that pairing. No training, ~40 min, every artefact already exists.
FUSION_TUNED="${FUSION_TUNED:-outputs/v4-fusion-tuned-embedder.json}"
STACKER_TUNED="${STACKER_TUNED:-outputs/v4-stacker-tuned.json}"
EMB_TUNED="${EMB_TUNED:-models2/embedding-tuned}"
IDX_TUNED="${IDX_TUNED:-artifacts2/dense-tuned}"

BLEND_TUNED() {
  "$PY" - "$FUSION_TUNED" "$@" <<'JSON'
import json, sys
from pathlib import Path
source = Path(sys.argv[1])
if not source.is_file():                      # fall back through what phase 2 left
    for name in ("outputs/v2-fusion-tuning-embedder.json", "outputs/v2-fusion-tuning.json"):
        if Path(name).is_file():
            source = Path(name)
            break
out = json.loads(source.read_text())["best"]
out["stacker"] = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
print(json.dumps(out))
JSON
}

stage_quick() {
  stage_now "quick"
  say "the untested cell: tuned embedder WITH a stacker (~40 min, no training)"

  command -v rclone >/dev/null && for item in \
      "models/embedding-tuned:$EMB_TUNED" "models/reranker-tuned-idx:models2/reranker-tuned-idx" \
      "dense-tuned:$IDX_TUNED"; do
    local remote="${item%%:*}" local_path="${item#*:}"
    [ -e "$local_path/config.json" ] || [ -e "$local_path/manifest.json" ] || {
      step "restoring $local_path from Drive"
      rclone copy "$RESULTS/$remote" "$local_path" -P --transfers 4 || true; }
  done
  [ -f "$STACKER_TUNED" ] || rclone copy "$RESULTS/outputs/$(basename "$STACKER_TUNED")" outputs -P 2>/dev/null || true

  local missing=0
  for f in "$EMB_TUNED/config.json" "$IDX_TUNED/manifest.json" "$STACKER_TUNED"; do
    [ -e "$f" ] || { step "MISSING $f"; missing=1; }
  done
  [ "$missing" = 0 ] || { step "cannot run without those - they are in Drive from phase 2/4"; return 1; }

  step "the stacker was fitted for SOME pipeline; if it was the base one its weights are"
  step "calibrated to the wrong score distribution and the 'refit' stage below is the fix."
  for reranker in models2/reranker-tuned-idx models2/reranker-g32; do
    [ -f "$reranker/config.json" ] || { step "skipping $reranker (not present)"; continue; }
    local tag; tag="l3q-$(basename "$reranker")"
    say "$(basename "$reranker") + tuned embedder + tuned stacker"
    if ! done_already "outputs/$tag.metrics.json"; then
      timed "predict $tag" "$PY" -m legalir2 predict --evaluate \
          --dense "$IDX_TUNED" --embedding-model "$EMB_TUNED" \
          --reranker-model "$reranker" \
          --settings "$(BLEND_TUNED "$STACKER_TUNED")" --output "outputs/$tag.json"
    fi
    "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
        --after "outputs/$tag.diagnostics.json" --label "$tag vs 0.9493" || true
    backup
  done

  gate "if either interval CLEARS ZERO you have the first real gain since 0.9493, and it
     cost forty minutes of a rented GPU and no training at all. If both span zero but
     the means are close to it, run 'refit' - the stacker may simply be calibrated to
     the wrong pipeline. If both are clearly negative, the reranker genuinely cannot
     exploit a better candidate pool and that is a finding worth writing down."
}

stage_refit() {
  stage_now "refit"
  say "refit the stacker ON the tuned pipeline, out of fold (~45 min)"
  step "v4-stacker-tuned.json may have been fitted against the base dense channel."
  step "a stacker reads per-question normalised retrieval features, so a different"
  step "dense channel changes every one of them."
  local reranker="${MODEL:-models2/reranker-tuned-idx}"
  [ -f "$reranker/config.json" ] || reranker=models2/reranker-g32
  timed "stack" "$PY" -m legalir2 stack --limit 2000 \
      --dense "$IDX_TUNED" --embedding-model "$EMB_TUNED" \
      --reranker-model "$reranker" --settings "$(BLEND_TUNED)" \
      --cache artifacts2/groups/stack-l3.jsonl --output outputs/l3-stacker.json

  timed "predict validation" "$PY" -m legalir2 predict --evaluate \
      --dense "$IDX_TUNED" --embedding-model "$EMB_TUNED" \
      --reranker-model "$reranker" \
      --settings "$(BLEND_TUNED outputs/l3-stacker.json)" \
      --output outputs/l3-refit.json
  "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/l3-refit.diagnostics.json --label "tuned pipeline, refit stacker"
  backup
}

stage_baseline() {
  stage_now "baseline"
  say "candidate recall as the system stands (~5 min)"
  step "this is the number legalir3 is trying to move. Final Recall@5 is downstream of"
  step "it and 58% as sensitive, which is why it is not the metric here."
  if done_already outputs/l3-baseline.json; then step "already measured"; else
    timed "retrieval" "$PY" -m legalir2 retrieval "${DENSE_NOW[@]}" \
        --settings "$(BLEND)" --output outputs/l3-baseline.json --label "legalir2 retrieval"
  fi
  backup
  gate "expect lexical ~0.85 @5, dense somewhere near it, fused 0.8914 @5 and 0.9805 @50.
     If fused @5 is not 0.8914 the settings do not match the shipped system and every
     later comparison is against the wrong thing."
}

stage_mine() {
  stage_now "mine"
  say "mine hard negatives from the DENSE channel's own ranking (~30 min)"
  step "the bi-encoder's negatives should be documents IT ranks highly and gets wrong."
  step "mining them from the fused order trains it against a distribution it never sees."
  if done_already "$BI_GROUPS"; then step "already mined; FORCE=1 to redo"; else
    timed "mine" "$PY" -m legalir2 mine "${DENSE_NOW[@]}" \
        --channel dense --depth 100 --negatives 15 --positives 2 \
        --settings "$(BLEND)" --output "$BI_GROUPS"
  fi
  "$PY" - <<'STATS'
import json
from pathlib import Path
p = Path("artifacts2/groups/bi-dense.stats.json")
s = json.loads(p.read_text())
print(f"  {s['questions']} groups, channel={s.get('channel')}, depth={s.get('depth')}")
print(f"  gold outside the dense top-{s.get('depth')}: {s.get('missed')} "
      f"({100 * s.get('missed', 0) / max(1, s['questions']):.1f}%)")
assert s.get("channel") == "dense", "stale miner - re-pull legalir2/mine.py"
STATS
  gate "the 'gold outside the dense top-100' figure is the dense channel's own recall
     miss, and it is the first time this project has measured it. Compare it against
     the fused miss in the baseline report: if it is much larger, the dense channel is
     carrying less of the fusion than assumed and this stage matters more, not less."
}

stage_train() {
  stage_now "train"
  [ -f "$BI_GROUPS" ] || { step "$BI_GROUPS missing - run mine first"; return 1; }
  say "fine-tune the bi-encoder on dense-channel negatives (~50 min)"
  step "InfoNCE, one positive against this question's hard negatives plus the batch."
  if done_already "$TUNED/config.json"; then step "already trained; FORCE=1 to redo"; else
    timed "train-bi" "$PY" -m legalir2 train-bi \
        --embedding-model "$EMBEDDER" --groups "$BI_GROUPS" \
        --output "$TUNED"
  fi
  backup
}

stage_index() {
  stage_now "index"
  [ -f "$TUNED/config.json" ] || { step "train first"; return 1; }
  say "rebuild the dense index with the tuned encoder (~1.2 h)"
  step "a dense index belongs to the encoder that built it; DenseIndex refuses a"
  step "mismatched one, so this cannot be skipped."
  if done_already "$INDEX/manifest.json"; then step "already indexed; FORCE=1 to redo"; else
    timed "index-dense" "$PY" -m legalir2 index-dense \
        --dense "$INDEX" --embedding-model "$TUNED"
  fi
  backup
}

stage_measure() {
  stage_now "measure"
  [ -f "$INDEX/manifest.json" ] || { step "index first"; return 1; }
  [ -f outputs/l3-baseline.json ] || { step "run baseline first"; return 1; }
  say "candidate recall with the tuned encoder, paired against the baseline (~5 min)"
  step "one evaluation pass; all three channels are compared from it. The dense channel"
  step "says whether the encoder improved, the fused channel says whether that survived"
  step "fusion - which is what the reranker actually receives."
  timed "retrieval" "$PY" -m legalir2 retrieval "${DENSE_NEW[@]}" \
      --settings "$(BLEND)" --output outputs/l3-tuned.json \
      --label "dense-channel negatives" --against outputs/l3-baseline.json
  backup
  gate "two intervals matter and they mean different things. @50 is the CEILING the
     reranker can reach - a gain there is headroom bought. @5 is the ORDER it starts
     from - a gain there is carried forward at 58%. Moving @50 without @5 is still
     progress; moving neither means the fix did not help and 'endtoend' is a waste."
}

stage_endtoend() {
  stage_now "endtoend"
  say "end to end, with the reranker and the shipped stacker (~20 min)"
  step "run this ONLY if measure showed a real retrieval gain. The conversion rate"
  step "means a 0.010 retrieval win lands here as 0.006, under the detection floor -"
  step "so a null here does NOT overturn a positive result upstream."
  local settings
  settings="$("$PY" -c "
import json,sys; s=json.loads(sys.argv[1]); s['stacker']='outputs/v4-stacker.json'
print(json.dumps(s))" "$(BLEND)")"
  timed "predict validation" "$PY" -m legalir2 predict --evaluate \
      --dense "$INDEX" --embedding-model "$TUNED" \
      --reranker-model models2/reranker-g32 --settings "$settings" \
      --output outputs/l3-endtoend.json
  "$PY" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/l3-endtoend.diagnostics.json --label "legalir3 retrieval vs 0.9493"
  "$PY" -m legalir2 audit --dense "$INDEX" --embedding-model "$TUNED" \
      --reranker-model models2/reranker-g32
  backup
  gate "the stacker here was fitted on the OLD dense channel's features. If the
     retrieval gain is real, refit it before drawing any conclusion from this number -
     that confound has already cost this project two experiments."
}

case "$STAGE" in
  quick)    stage_quick ;;
  refit)    stage_refit ;;
  baseline) stage_baseline ;;
  mine)     stage_mine ;;
  train)    stage_train ;;
  index)    stage_index ;;
  measure)  stage_measure ;;
  endtoend) stage_endtoend ;;
  all)      stage_baseline; stage_mine; stage_train; stage_index; stage_measure ;;
  *)        echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
