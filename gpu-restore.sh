#!/usr/bin/env bash
# Rebuild a working box from Google Drive after the previous rental was terminated.
#
#   bash gpu-restore.sh
#
# Pulls the code and corpus, restores whatever trained weights survived, rebuilds
# everything that is cheap to rebuild, and then tells you exactly which stage to run
# next. Safe to re-run.
#
# Assumes rclone is installed and ~/.config/rclone/rclone.conf holds the gdrive remote.
set -euo pipefail

ROOT="${ROOT:-/workspace/DSC-2026}"
REMOTE="${REMOTE:-gdrive:DSC-2026}"
RESULTS="${RESULTS:-gdrive:DSC-2026-results}"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

command -v rclone >/dev/null || { echo "install rclone first: curl https://rclone.org/install.sh | sudo bash"; exit 1; }

say "pulling code and corpus"
mkdir -p "$ROOT"
rclone copy "$REMOTE" "$ROOT" --progress
cd "$ROOT"
[ -f corpus.jsonl.gz ] || { echo "corpus.jsonl.gz missing from $REMOTE - upload _upload again"; exit 1; }

say "checking what survived the last rental"
HAVE_RERANKER=0; HAVE_EMBEDDER=0; HAVE_GROUPS=0
if rclone lsf "$RESULTS" >/dev/null 2>&1; then
  LISTING="$(rclone lsf -R "$RESULTS" 2>/dev/null || true)"
  echo "$LISTING" | grep -q "models/reranker-tuned/"   && HAVE_RERANKER=1
  echo "$LISTING" | grep -q "models/embedding-tuned/"  && HAVE_EMBEDDER=1
  echo "$LISTING" | grep -q "groups/train.jsonl"       && HAVE_GROUPS=1
  step "restoring what is there (weights are ~2.3 GB each, this takes a few minutes)"
  mkdir -p models2 artifacts2/groups outputs
  [ "$HAVE_RERANKER" = 1 ] && rclone copy "$RESULTS/models/reranker-tuned"  models2/reranker-tuned  -P
  [ "$HAVE_EMBEDDER" = 1 ] && rclone copy "$RESULTS/models/embedding-tuned" models2/embedding-tuned -P
  [ "$HAVE_GROUPS"   = 1 ] && rclone copy "$RESULTS/groups" artifacts2/groups -P
  rclone copy "$RESULTS/outputs" outputs -P 2>/dev/null || true
else
  step "no results folder in Drive - nothing survived"
fi

# Checkpoints trained before the provenance fix carry the base model's hashes, which
# makes verify-models call good weights corrupt. Retire that file rather than trust it.
for d in models2/reranker-tuned models2/embedding-tuned models2/reranker-g32; do
  if [ -f "$d/provenance.json" ] && ! grep -q derived_from "$d/provenance.json" 2>/dev/null \
     && grep -q '"files"' "$d/provenance.json" 2>/dev/null; then
    step "retiring stale base provenance in $d (it describes the untrained weights)"
    mv "$d/provenance.json" "$d/provenance.base.json"
  fi
done

printf '\n  trained reranker : %s\n' "$([ "$HAVE_RERANKER" = 1 ] && echo 'restored' || echo 'LOST - needs ~2 h to retrain')"
printf '  tuned embedder   : %s\n'   "$([ "$HAVE_EMBEDDER" = 1 ] && echo 'restored' || echo 'not trained yet')"
printf '  mined groups     : %s\n'   "$([ "$HAVE_GROUPS"   = 1 ] && echo 'restored' || echo 'missing - 11 min to re-mine')"

say "rebuilding the cheap things (environment, store, postings)"
bash gpu-setup.sh env
bash gpu-setup.sh index

say "what to run next"
if [ "$HAVE_RERANKER" = 1 ]; then
  cat <<'NEXT'
  Your 0.9223 reranker is back. Skip straight to phase two:

      bash gpu-phase2.sh dense

  ~2 h, and it ends at the gate that decides whether 0.96 is reachable.
NEXT
else
  cat <<'NEXT'
  The trained reranker did not survive. Rebuild it first - but take the improved
  recipe rather than repeating the old one, since it costs the same and should
  land higher:

      python -m legalir2 download --which reranker && python -m legalir2 verify-models
      python -m legalir2 mine --no-dense --negatives 31
      python -m legalir2 train-ce --group-size 32 --groups-per-step 1 --accumulate 4 \
          --epochs 2 --output models2/reranker-tuned
      python -m legalir2 predict --no-dense --evaluate \
          --reranker-model models2/reranker-tuned --output outputs/v2-ce-tuned.json

  Compare against 0.8483 (lexical) and the 0.9223 you got before, then:

      bash gpu-phase2.sh dense

  And upload as soon as it finishes this time:
      rclone copy /workspace/DSC-2026/_results gdrive:DSC-2026-results -P
NEXT
fi
