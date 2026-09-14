#!/usr/bin/env bash
# Phase two: 0.9223 -> the 0.96 attempt. Run on the SAME box as gpu-setup.sh.
#
#   bash gpu-phase2.sh dense      # ~2 h  embedder + dense index + tuned fusion + measure
#   bash gpu-phase2.sh rerank     # ~2 h  re-mine with more negatives, retrain at group size 32
#   bash gpu-phase2.sh biencoder  # ~3 h  fine-tune the embedder, re-index, re-tune fusion
#   bash gpu-phase2.sh final      # ~1 h  retrain on all 7,000, predict, pack
#   bash gpu-phase2.sh            # all of the above, in order
#
# Each stage ends with a measured number and a GATE telling you whether the next
# stage is worth its GPU hours. Stop at a failed gate rather than pushing through.
#
#   tmux new -s p2  ->  bash gpu-phase2.sh  ->  detach ctrl-b d
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs
PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
export HF_HUB_DISABLE_TELEMETRY=1

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
gate() { printf '\n>>> GATE: %s\n\n' "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }
timed() { local l="$1"; shift; step "$l ..."; local t=$SECONDS; "$@"; step "$l finished in $((SECONDS-t))s"; }

# Measuring the validation split costs ~16 min of GPU time and is deterministic: on a
# re-run the answer is already on disk, and re-deriving it changes nothing except how
# long you wait to resume. A finished measurement is therefore skipped exactly like a
# finished index or checkpoint, and its recorded number is printed instead. Without
# this, "safe to re-run" quietly costs an hour across the four stages - which is the
# difference between resuming being free and resuming being something you avoid.
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

# Fusion settings chosen by `tune --stage fusion`, as a --settings JSON string.
#
# Each embedder gets its OWN file. They used to share one, which meant the biencoder
# stage overwrote the base embedder's weights with weights fitted against the tuned
# index - harmless in a single forward pass, wrong the moment anything re-measured the
# base index afterwards, which resuming makes likely. Two files, no aliasing.
BASE_FUSION="outputs/v2-fusion-tuning.json"
TUNED_FUSION="outputs/v2-fusion-tuning-embedder.json"
fusion_settings() {
  local file="${1:-$BASE_FUSION}"
  if [ -f "$file" ]; then
    "$PY" -c "import json;b=json.load(open('$file'))['best'];print(json.dumps(b))"
  else
    echo '{}'
  fi
}

stage_dense() {
  say "embedder checkpoint"
  if done_already models2/embedding/config.json; then
    step "already downloaded"
  else
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
    timed "download embedder" "$PY" -m legalir2 download --which embedding
    timed "verify hashes" "$PY" -m legalir2 verify-models
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  say "index every chunk with the embedder (~1.2 h)"
  if done_already artifacts2/dense/manifest.json; then
    step "already indexed; FORCE=1 to rebuild"
  else
    timed "index-dense" "$PY" -m legalir2 index-dense --batch-size 32
  fi

  say "tune how the two channels fuse (~15 min, mostly the dense forward pass)"
  step "not optional: after in-domain tuning, naive fusion degrades more often than it helps"
  if done_already "$BASE_FUSION"; then
    step "already tuned: $(fusion_settings)"
  else
    timed "tune fusion" "$PY" -m legalir2 tune --stage fusion --limit 400 --target-depth 100 \
        --output "$BASE_FUSION"
  fi

  say "measure: dense added, reranker unchanged"
  measure "predict validation" outputs/v2-dense-ce.json \
      --reranker-model models2/reranker-tuned \
      --settings "$(fusion_settings)"
  "$PY" - <<'REPORT'
import json
m = json.load(open("outputs/v2-dense-ce.metrics.json"))
c = m["candidate_recall"]
print(f"\n  Recall@5 {m['recall']:.4f}   candidate recall @50 {c['@50']} @100 {c['@100']}")
print(f"  lexical-only was: Recall 0.8483 (no rerank), candidate @50 0.9608 @100 0.9752")
ceiling = float(c["@100"])
print(f"\n  reranker conversion = {m['recall']/ceiling:.1%} of candidate recall@100")
print(f"  for a final 0.96 you need candidate recall >= 0.985; you are at {ceiling}")
REPORT
  gate "candidate recall @100 should now be >= 0.98. If it is still under 0.975, the
     embedder is the wrong checkpoint - run 'probe' across two or three alternatives
     from the sheet before spending three hours fine-tuning this one."
}

stage_rerank() {
  # The base checkpoint is what fine-tuning starts from, and gpu-restore.sh only brings
  # back the *trained* weights. Check it here rather than after 11 minutes of mining -
  # transformers reports a missing local path as an HFValidationError about repo ids,
  # which sends you looking for a network problem that does not exist.
  if [ ! -f models2/reranker/config.json ]; then
    step "base reranker missing - fetching it before anything expensive runs"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
    timed "download reranker" "$PY" -m legalir2 download --which reranker
    timed "verify hashes" "$PY" -m legalir2 verify-models
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  say "re-mine with many more negatives, now using both channels"
  step "group size 8 -> 32 is a continuous gain in the literature, and costs no extra VRAM"
  if done_already artifacts2/groups/train-r2.jsonl; then
    step "already mined"
  else
    timed "mine round two" "$PY" -m legalir2 mine --negatives 31 \
        --settings "$(fusion_settings)" \
        --output artifacts2/groups/train-r2.jsonl
  fi
  cat artifacts2/groups/train-r2.stats.json
  "$PY" - <<'CHECK'
import json, sys
stats = json.load(open("artifacts2/groups/train-r2.stats.json"))
if "fusion" not in stats:
    sys.exit("\n  STOP: these groups were mined by the old miner, which used its own\n"
             "  blend of the two channels instead of the ranking prediction uses.\n"
             "  Training on them repeats the mistake. Delete and re-mine:\n"
             "      rm artifacts2/groups/train-r2.*\n")
print(f"\n  mined through the prediction path, fusion = {stats['fusion']}, "
      f"dense = {stats['dense']}")
CHECK

  say "retrain the cross encoder at group size 32 (~2 h)"
  step "32 x 1 per step = the same 32 pairs per forward as 8 x 4, so VRAM is unchanged"
  if done_already models2/reranker-g32/config.json; then
    step "already trained; FORCE=1 to retrain"
  else
    timed "train-ce" "$PY" -m legalir2 train-ce \
        --groups artifacts2/groups/train-r2.jsonl \
        --group-size 32 --groups-per-step 1 --accumulate 4 --epochs 2 \
        --output models2/reranker-g32
  fi

  say "measure"
  measure "predict validation" outputs/v2-g32.json \
      --reranker-model models2/reranker-g32 \
      --settings "$(fusion_settings)"
  compare outputs/v2-dense-ce.metrics.json outputs/v2-g32.metrics.json "group size 32"

  say "ensemble the two rerankers - they disagree on different questions, and the"
  step "old one is already paid for. Wide pass by g32, top 25 rescored by both."
  measure "predict validation" outputs/v2-ens.json \
      --reranker-model models2/reranker-g32 models2/reranker-tuned \
      --settings "$(fusion_settings)"
  compare outputs/v2-g32.metrics.json outputs/v2-ens.metrics.json "+ ensemble"
  "$PY" -m legalir2 audit --reranker-model models2/reranker-g32 models2/reranker-tuned \
      2>/dev/null || step "run 'audit' yourself before submitting an ensemble - two "\
"rerankers plus the embedder must still total under 4B parameters"
  gate "this should be the largest single jump of phase two. If it is flat or worse,
     suspect the mined negatives: read train-r2.stats.json and compare 'missed' and
     'pruned' against the round-one file before training anything else."
}

stage_biencoder() {
  say "fine-tune the embedder on the same labels (~1.5 h)"
  if done_already models2/embedding-tuned/config.json; then
    step "already trained"
  else
    timed "train-bi" "$PY" -m legalir2 train-bi \
        --groups artifacts2/groups/train-r2.jsonl \
        --epochs 2 --batch-size 8 --negatives 7 --output models2/embedding-tuned
  fi

  say "re-index with the tuned embedder (~1.2 h)"
  if done_already artifacts2/dense-tuned/manifest.json; then
    step "already indexed"
  else
    timed "index-dense" "$PY" -m legalir2 index-dense \
        --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
        --batch-size 32
  fi

  say "re-tune fusion - the old weights were fitted against the old embedder"
  if done_already "$TUNED_FUSION"; then
    step "already tuned: $(fusion_settings "$TUNED_FUSION")"
  else
    timed "tune fusion" "$PY" -m legalir2 tune --stage fusion --limit 400 --target-depth 100 \
        --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
        --output "$TUNED_FUSION"
  fi

  say "measure"
  measure "predict validation" outputs/v2-bi-tuned.json \
      --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
      --reranker-model models2/reranker-g32 \
      --settings "$(fusion_settings "$TUNED_FUSION")"
  compare outputs/v2-g32.metrics.json outputs/v2-bi-tuned.metrics.json "tuned bi-encoder"
  gate "if candidate recall @100 is now >= 0.985, 0.96 is live. If it stalled near 0.975,
     the remaining misses are probably label noise rather than retrieval failures -
     read the diagnostics before spending more."
}

stage_final() {
  say "deeper shortlist, two-stage rerank"
  measure "predict validation" outputs/v2-deep.json \
      --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
      --reranker-model models2/reranker-g32 \
      --settings "$("$PY" -c "
import json, os
path = '$TUNED_FUSION' if os.path.isfile('$TUNED_FUSION') else '$BASE_FUSION'
b = json.load(open(path))['best']
b.update(candidates=100, passages=4); print(json.dumps(b))")"
  compare outputs/v2-bi-tuned.metrics.json outputs/v2-deep.metrics.json "candidates=100 passages=4"

  say "pick the best configuration and submit it"
  "$PY" - <<'PICK'
import glob, json
rows = []
for path in sorted(glob.glob("outputs/v2-*.metrics.json")):
    try:
        m = json.load(open(path))
        rows.append((m["recall"], path, m.get("candidate_recall", {})))
    except Exception:
        pass
rows.sort(reverse=True)
print("\n  validation Recall@5, best first:")
for recall, path, cand in rows:
    print(f"   {recall:.4f}  {path.split('/')[-1]:<32} candidates@100={cand.get('@100','-')}")
print("\n  Use the best row's model and settings for the public run.")
PICK
  gate "before the all-7,000 retrain: that step puts the validation questions inside
     training, so this table is the last honest comparison you will get. Write it down."
}

compare() {
  "$PY" - "$1" "$2" "$3" <<'CMP'
import json, sys
before, after, label = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
delta = after["recall"] - before["recall"]
arrow = "up" if delta > 0 else "DOWN"
print(f"\n  {label}: Recall@5 {before['recall']:.4f} -> {after['recall']:.4f}  ({arrow} {delta:+.4f})")
for depth in ("@50", "@100"):
    b = before.get("candidate_recall", {}).get(depth)
    a = after.get("candidate_recall", {}).get(depth)
    if b and a:
        print(f"  candidate recall {depth}: {b} -> {a}")
CMP
}

case "$STAGE" in
  dense)     stage_dense ;;
  rerank)    stage_rerank ;;
  biencoder) stage_biencoder ;;
  final)     stage_final ;;
  all)       stage_dense; stage_rerank; stage_biencoder; stage_final ;;
  *)         echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
