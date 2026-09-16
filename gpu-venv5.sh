#!/usr/bin/env bash
# A SECOND interpreter on transformers 5.x, so the 2B can be tried without touching
# the environment that produced the 0.9493 submission.
#
#   bash gpu-venv5.sh            # build .venv5, then run the three checks
#
# WHY NOT JUST UPGRADE
# --------------------
# Prism-Qwen3.5-Reranker-2B has model_type `qwen3_5_text`, which transformers only
# learned in 5.2. The box is on 4.51.3. Upgrading in place would move the embedder,
# the 0.6B reranker and every tokeniser onto a new major version at the same time.
#
# The danger is not a crash - a crash would be fine, it announces itself. The danger
# is that everything loads and something small changes: a tokeniser default, a pooling
# path, an attention implementation. Then `measure` compares a 2B model under
# transformers 5 against v4-stacked.diagnostics.json, produced under 4.51.3, and
# reports the sum of two changes as if it were one.
#
# A second venv costs a few minutes of disk and keeps .venv byte-for-byte the system
# that scored 0.9493. If 5.x turns out to be fine, the cost was the disk. If it is not,
# the cost was still only the disk.
#
# WHAT THIS DOES NOT SETTLE
# -------------------------
# Even if all three checks pass, phase six run under .venv5 is not yet comparable to a
# baseline measured under .venv. Check 3 is what closes that gap: it re-measures the
# SHIPPED 0.6B pipeline under transformers 5 and diffs it against the stored
# diagnostics. Identical means the two environments rank the same way and the
# comparison is clean. Different means every later number is confounded, and the answer
# is to re-baseline under .venv5 rather than to hope.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV5="${VENV5:-$ROOT/.venv5}"
PY5="$VENV5/bin/python"
TRANSFORMERS="${TRANSFORMERS:-transformers>=5.2,<6}"
MODEL="${MODEL:-models2/reranker2b}"
STAGE="${1:-all}"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

stage_build() {
  say "building $VENV5"
  if [ -x "$PY5" ]; then
    step "already exists - delete $VENV5 to rebuild"
  else
    # --system-site-packages inherits the CUDA torch that is already on the box.
    # A pip install INTO the venv shadows a system package of the same name, so
    # transformers 5 lands here and 4.51.3 stays visible to .venv untouched.
    step "inheriting system torch rather than downloading 2.5 GB again"
    python3 -m venv --system-site-packages "$VENV5"
    "$PY5" -m pip install -q --upgrade pip
  fi

  if ! "$PY5" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    step "no CUDA torch inherited - installing one into $VENV5 (~2.5 GB)"
    "$PY5" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  fi

  step "installing $TRANSFORMERS (this is the whole point of the second venv)"
  "$PY5" -m pip install -q --upgrade "$TRANSFORMERS" peft sentencepiece protobuf "numpy>=1.26"

  "$PY5" - <<'CHECK'
import torch, transformers
print(f"  .venv5: transformers {transformers.__version__}, torch {torch.__version__}")
print(f"  cuda {torch.cuda.is_available()}  {torch.cuda.get_device_name(0)}")
CHECK
  step ".venv (untouched): transformers $("$ROOT/.venv/bin/python" -c 'import transformers; print(transformers.__version__)')"
}

stage_arch() {
  say "1. does transformers 5 know qwen3_5_text?"
  "$PY5" - "$MODEL" <<'ARCH'
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
print(f"  config class : {type(config).__name__}")
print(f"  model_type   : {getattr(config, 'model_type', '?')}")
ARCH
  step "if that printed a class name, the upgrade was the whole problem"
}

stage_tests() {
  say "2. does this project still work on transformers 5?"
  step "97 tests. A failure here is a real incompatibility, not a version warning."
  if "$PY5" -m unittest discover -s tests2 -t . > outputs/tests-venv5.log 2>&1; then
    tail -3 outputs/tests-venv5.log
  else
    echo "  FAILED under transformers 5 - full output in outputs/tests-venv5.log:"
    tail -40 outputs/tests-venv5.log
    exit 1
  fi
}

stage_baseline() {
  say "3. does the SHIPPED pipeline rank identically under transformers 5? (~35 min)"
  step "this is the check that makes a later comparison meaningful. It re-runs the"
  step "0.6B system that scored 0.9493 - same weights, same settings, new library -"
  step "and diffs the result against the stored diagnostics."
  [ -f outputs/v4-stacked.diagnostics.json ] || {
    step "outputs/v4-stacked.diagnostics.json missing - rclone copy the outputs folder"; return 1; }

  local settings
  settings="$("$ROOT/.venv/bin/python" - <<'JSON'
import json
from pathlib import Path
out = json.loads(Path("outputs/v2-fusion-tuning.json").read_text())["best"]
out["stacker"] = "outputs/v4-stacker.json"
print(json.dumps(out))
JSON
)"
  "$PY5" -m legalir2 predict --evaluate \
      --dense artifacts2/dense --embedding-model models2/embedding \
      --reranker-model models2/reranker-g32 --settings "$settings" \
      --output outputs/v5-baseline-tf5.json

  say "transformers 4.51.3 vs transformers 5 - same model, same settings"
  "$PY5" -m legalir2 compare --before outputs/v4-stacked.diagnostics.json \
      --after outputs/v5-baseline-tf5.diagnostics.json \
      --label "library upgrade only"

  cat <<'VERDICT'

  READ THAT INTERVAL DIFFERENTLY THAN USUAL.

  Everywhere else in this project, an interval spanning zero is a disappointment.
  Here it is the PASS: nothing changed except the library, so nothing SHOULD have
  changed, and a difference of +-0.000 means the two environments rank identically
  and phase six can be compared against the stored baseline.

  A difference that clears zero in either direction means transformers 5 altered the
  0.6B pipeline. Then v4-stacked is no longer a valid baseline for anything measured
  under .venv5, and the fix is to use v5-baseline-tf5.diagnostics.json as the
  comparison point instead - measuring the 2B against the 0.6B under the SAME library.
VERDICT
}

case "$STAGE" in
  build)    stage_build ;;
  arch)     stage_arch ;;
  tests)    stage_tests ;;
  baseline) stage_baseline ;;
  all)      stage_build; stage_arch; stage_tests
            printf '\n  Checks 1 and 2 passed. Check 3 costs ~35 min and is what makes\n'
            printf '  the phase-six comparison trustworthy:\n\n'
            printf '      bash gpu-venv5.sh baseline\n\n' ;;
  *)        echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
