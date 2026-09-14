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
  echo "$LISTING" | grep -q "models/embedding-tuned/"  && HAVE_EMBEDDER=1
  echo "$LISTING" | grep -q "groups/train"             && HAVE_GROUPS=1
  step "restoring what is there (weights are ~1.1 GB each, this takes a few minutes)"
  mkdir -p models2 artifacts2/groups outputs

  # Restore EVERY model directory in Drive, not a hardcoded list. The list was
  # hardcoded to reranker-tuned, so reranker-g32 - the checkpoint behind the 0.9433
  # submission and the base every later phase builds on - was silently left behind on
  # a fresh box. Discovering what is there beats maintaining a list that goes stale
  # every time a phase adds a checkpoint.
  for path in $(echo "$LISTING" | grep "^models/[^/]*/config.json$" | sed 's#^models/##; s#/config.json$##'); do
    step "restoring models2/$path"
    rclone copy "$RESULTS/models/$path" "models2/$path" -P --transfers 4 || true
    [ "$path" = "reranker-g32" ] && HAVE_RERANKER=1
    [ "$path" = "reranker-tuned" ] && [ "$HAVE_RERANKER" = 0 ] && HAVE_RERANKER=1
  done
  [ "$HAVE_GROUPS"   = 1 ] && rclone copy "$RESULTS/groups" artifacts2/groups -P
  # A restored dense index saves 1.2 h of GPU time per embedder. It is only valid for
  # the store it was built from, and DenseIndex refuses to load a mismatched one, so a
  # stale copy fails loudly at load time rather than quietly scoring against the wrong
  # chunk ids - which makes restoring it the safe default rather than a gamble.
  for d in dense dense-tuned; do
    if echo "$LISTING" | grep -q "$d/manifest.json"; then
      step "restoring artifacts2/$d (~700 MB, saves 1.2 h of indexing)"
      rclone copy "$RESULTS/$d" "artifacts2/$d" -P --transfers 4 || true
    fi
  done
  rclone copy "$RESULTS/outputs" outputs -P 2>/dev/null || true
else
  step "no results folder in Drive - nothing survived"
fi

# Checkpoints trained before the provenance fix carry the base model's hashes, which
# makes verify-models call good weights corrupt. Retire that file rather than trust it.
for d in models2/*/; do
  d="${d%/}"
  if [ -f "$d/provenance.json" ] && ! grep -q derived_from "$d/provenance.json" 2>/dev/null \
     && grep -q '"files"' "$d/provenance.json" 2>/dev/null; then
    step "retiring stale base provenance in $d (it describes the untrained weights)"
    mv "$d/provenance.json" "$d/provenance.base.json"
  fi
done

printf '\n  restored checkpoints:\n'
for d in models2/*/; do
  [ -f "${d}config.json" ] && printf '    %-22s %s\n' "$(basename "${d%/}")" \
      "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
printf '\n  trained reranker : %s\n' "$([ "$HAVE_RERANKER" = 1 ] && echo 'restored' || echo 'LOST - needs ~2 h to retrain')"
printf '  tuned embedder   : %s\n'   "$([ "$HAVE_EMBEDDER" = 1 ] && echo 'restored' || echo 'not trained yet')"
printf '  mined groups     : %s\n'   "$([ "$HAVE_GROUPS"   = 1 ] && echo 'restored' || echo 'missing - 11 min to re-mine')"
printf '  dense index      : %s\n'   "$([ -f artifacts2/dense/manifest.json ] && echo 'restored' || echo 'missing - 1.2 h to rebuild')"
printf '  tuned dense index: %s\n'   "$([ -f artifacts2/dense-tuned/manifest.json ] && echo 'restored' || echo 'not built yet')"

say "rebuilding the cheap things (environment, store, postings)"
bash gpu-setup.sh env
bash gpu-setup.sh index

# Restoring a fine-tuned checkpoint is not the same as having the base one: every
# later training run starts from the base, and its absence only surfaces as an
# HFValidationError about repo ids once the expensive steps have already finished.
say "base checkpoints (fine-tuning starts from these, restore does not bring them back)"
for which in reranker embedding; do
  if [ -f "models2/$which/config.json" ]; then
    step "$which base already present"
  else
    step "fetching $which base"
    ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      .venv/bin/python -m legalir2 download --which "$which" )
  fi
done

say "what to run next"
if [ "$HAVE_RERANKER" = 1 ]; then
  cat <<'NEXT'
  Your trained rerankers are back. Phase four is the current plan:

      tmux new -s p4        # from OUTSIDE tmux
      bash gpu-auto4.sh

  ~9 h unattended: multi-positive loss, learned rank fusion, the rerank sweep, then
  the ceiling stage. Logs ship to Drive every 60 s:

      rclone cat gdrive:DSC-2026-results/logs/STATUS.txt
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
