#!/usr/bin/env bash
# legalir3, stage one: fix the retrieval half.
#
#   bash gpu-legalir3.sh baseline   # ~20 min  candidate recall as it stands today
#   bash gpu-legalir3.sh mine       # ~30 min  negatives from the DENSE channel
#   bash gpu-legalir3.sh train      # ~2 h     fine-tune the bi-encoder on them
#   bash gpu-legalir3.sh index      # ~1.2 h   rebuild the dense index
#   bash gpu-legalir3.sh measure    # ~25 min  paired candidate recall vs baseline
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

stage_baseline() {
  stage_now "baseline"
  say "candidate recall as the system stands (~20 min)"
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
  say "fine-tune the bi-encoder on dense-channel negatives (~2 h)"
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
  say "candidate recall with the tuned encoder, paired against the baseline (~25 min)"
  timed "retrieval" "$PY" -m legalir2 retrieval "${DENSE_NEW[@]}" \
      --settings "$(BLEND)" --output outputs/l3-tuned.json \
      --label "dense-channel negatives" --against outputs/l3-baseline.json --channel dense

  say "and what it did to the fusion, which is what the reranker actually receives"
  "$PY" -m legalir2 retrieval "${DENSE_NEW[@]}" --settings "$(BLEND)" \
      --output outputs/l3-tuned.json --label "fusion" \
      --against outputs/l3-baseline.json --channel fused
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
  baseline) stage_baseline ;;
  mine)     stage_mine ;;
  train)    stage_train ;;
  index)    stage_index ;;
  measure)  stage_measure ;;
  endtoend) stage_endtoend ;;
  all)      stage_baseline; stage_mine; stage_train; stage_index; stage_measure ;;
  *)        echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
