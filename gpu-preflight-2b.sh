#!/usr/bin/env bash
# Prism-Qwen3.5-Reranker-2B preflight: ~20 minutes to find out if the ~8-hour version
# is worth starting. Run this BEFORE any cross-fit or caching with the 2B model.
#
#   bash gpu-preflight-2b.sh
#
# Three things can go wrong, and only one of them announces itself:
#
#   1. transformers does not know the Qwen3.5 architecture -> KeyError on load. Loud,
#      and the cheapest possible failure, which is why it is checked first.
#   2. The prompt template is wrong -> NO error. Every score is quietly worse. This is
#      why models.py now carries a per-checkpoint template, and why this script scores
#      a known-relevant passage against a known-irrelevant one and asserts the ordering.
#   3. It does not fit, or is too slow to be usable -> measured here, in q/s, so the
#      decision to spend eight hours is made against a number rather than a hope.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-models2/reranker2b}"
BATCH="${BATCH:-32}"
export HF_HUB_DISABLE_TELEMETRY=1

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "1. download (~4 GB)"
if [ -f "$MODEL/config.json" ]; then
  step "already present"
else
  ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    "$PY" -m legalir2 download --which reranker2b )
fi
"$PY" -c "
import json
from pathlib import Path
config = json.loads(Path('$MODEL/config.json').read_text())
print(f\"  model_type       : {config.get('model_type')}\")
print(f\"  layers / hidden  : {config.get('num_hidden_layers')} / {config.get('hidden_size')}\")
print(f\"  max positions    : {config.get('max_position_embeddings')}\")"

say "2. does this transformers understand the architecture?"
step "a KeyError here is the whole answer - stop and upgrade transformers"
"$PY" -c "
import transformers
from transformers import AutoConfig
print(f'  transformers {transformers.__version__}')
c = AutoConfig.from_pretrained('$MODEL', local_files_only=True)
print(f'  config class     : {type(c).__name__}')"

say "3. parameter count and the deployed budget"
"$PY" -m legalir2 audit --reranker-model "$MODEL" \
    --dense artifacts2/dense --embedding-model models2/embedding

say "4. does it actually rank? (the silent-failure check)"
step "a known-relevant passage must outscore a known-irrelevant one. If this fails,"
step "the prompt template is wrong - and nothing else would have told you."
"$PY" - "$MODEL" <<'PROBE'
import sys, time
from legalir2.rerank import Reranker
from legalir2.store import Store

model = Reranker(sys.argv[1], device="cuda", max_length=768)
print(f"  template system  : {model.template['system'][:60]}...")
print(f"  yes/no token ids : {model.labels}")

store = Store("artifacts2/store.sqlite")
gold = store.doc_ids[0]
chunks = store.document_chunk_ids(gold)
relevant = store.chunk_text(chunks[0], with_label=False)
query = " ".join(relevant.split()[:15])
# something from a different document entirely
other = next(d for d in store.doc_ids[1:] if store.document_chunk_ids(d))
irrelevant = store.chunk_text(store.document_chunk_ids(other)[0], with_label=False)
store.close()

scores = model.score(query, [relevant, irrelevant], batch_size=2, max_length=768)
print(f"  relevant   {scores[0]:+.4f}")
print(f"  irrelevant {scores[1]:+.4f}")
if scores[0] <= scores[1]:
    raise SystemExit(
        "\n  FAILED: the irrelevant passage scored at least as high as the relevant one.\n"
        "  Do NOT proceed - check the prompt template against the model card before\n"
        "  spending eight hours training and caching against a broken scorer.\n")
print("  ordering correct\n")

passages = [relevant] * 64
started = time.time()
model.score(query, passages, batch_size=int(len(passages) / 2), max_length=768)
rate = len(passages) / (time.time() - started)
print(f"  throughput       : {rate:.0f} passages/s")
print(f"  a 3,500-question fold at 100 docs x 3 passages would take "
      f"{3500 * 300 / rate / 3600:.1f} h")
model.close()
PROBE

say "verdict"
cat <<'DONE'
  If every section above passed, the 2B path is viable and the caching estimate is
  real rather than assumed. Compare that hours figure against what the 0.6B took
  before committing: the 2B is roughly 3x slower per passage, and every future
  measurement pays that tax for as long as it ships.

  If section 4 failed, the template is wrong. Fix models.py TEMPLATES["prism"]
  against the model card and re-run this - it costs twenty minutes, not eight hours.
DONE
