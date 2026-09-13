#!/usr/bin/env bash
# Bring a rented GPU box from an empty directory to a packed submission.
#
#   bash gpu-setup.sh            # everything: env, index, mine, train, predict, pack
#   bash gpu-setup.sh env        # just the environment
#   bash gpu-setup.sh index      # store + lexical postings (CPU, ~5 min)
#   bash gpu-setup.sh train      # mine + train the cross encoder
#   bash gpu-setup.sh submit     # measure on validation, predict public, pack
#   bash gpu-setup.sh results    # collect outputs and weights into _results/
#
# Run it under tmux: an SSH drop kills a foreground job, and training is hours long.
#   tmux new -s dsc  ->  bash gpu-setup.sh  ->  detach ctrl-b d, return with tmux a
#
# Re-running any stage is safe: finished stages are skipped unless FORCE=1.
# Every slow step announces itself and reports its duration, because silence during
# a multi-gigabyte download is indistinguishable from a hang.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p outputs
PY="${PY:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
FORCE="${FORCE:-0}"

say()  { printf '\n=== %s ===\n' "$*"; }
step() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
done_already() { [ "$FORCE" = "0" ] && [ -e "$1" ]; }

timed() {
  local label="$1"; shift
  step "$label ..."
  local t0=$SECONDS
  "$@"
  step "$label finished in $((SECONDS - t0))s"
}

stage_env() {
  say "environment"
  step "python $(python3 -V 2>&1 | cut -d' ' -f2), $(nproc) vCPU, $(free -g | awk '/^Mem:/{print $2}') GB RAM, $(df -h "$ROOT" | awk 'NR==2{print $4}') free disk"
  step "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"

  if [ ! -x "$PY" ]; then
    # Importing torch on a cold network filesystem takes up to a minute on its own,
    # and cuda.is_available() initialises a CUDA context. Say so before waiting.
    step "checking for a CUDA torch in the image (first import can take ~60s) ..."
    if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      step "found one - reusing it instead of downloading ~2.5 GB"
      timed "create venv (system site packages)" \
        python3 -m venv --system-site-packages "$ROOT/.venv"
    else
      step "none found - torch will be downloaded (~2.5 GB, several minutes)"
      timed "create venv" python3 -m venv "$ROOT/.venv"
    fi
    timed "upgrade pip" "$ROOT/.venv/bin/pip" install -q --upgrade pip
    if ! "$ROOT/.venv/bin/python" -c "import torch" 2>/dev/null; then
      # Deliberately not -q: this is the long pole and you want the progress bar.
      timed "install CUDA torch" "$ROOT/.venv/bin/pip" install \
        torch --index-url https://download.pytorch.org/whl/cu128
    fi
    # sentencepiece and protobuf are not optional here: the approved Vietnamese
    # checkpoints ship a SentencePiece model rather than a prebuilt tokenizer.json,
    # so transformers takes the slow-tokenizer conversion path, which needs both.
    timed "install transformers, numpy, tokenizer deps" "$ROOT/.venv/bin/pip" install \
      "transformers==4.51.3" "numpy>=1.26" sentencepiece protobuf
  else
    step "reusing existing interpreter: $PY"
  fi

  "$PY" - <<'CHECK'
import importlib, torch
missing = [m for m in ("sentencepiece", "google.protobuf")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "missing tokenizer dependencies: " + ", ".join(missing) +
        "\n  fix with: .venv/bin/pip install sentencepiece protobuf")
assert torch.cuda.is_available(), "no CUDA device visible"
total = torch.cuda.mem_get_info()[1]
print(f"  {torch.cuda.get_device_name(0)}  {total/2**30:.0f} GiB  torch {torch.__version__}  "
      f"bf16={'yes' if torch.cuda.is_bf16_supported() else 'NO - will fall back to fp16'}")
if total / 2**30 < 15:
    print("  warning: under 16 GiB. train-ce needs ~12 GiB, train-bi ~14 GiB.")
CHECK

  step "running the test suite ..."
  if "$PY" -m unittest discover -s tests2 -t . > outputs/tests.log 2>&1; then
    tail -3 outputs/tests.log
  else
    echo "TESTS FAILED - full output is in outputs/tests.log:"
    tail -40 outputs/tests.log
    exit 1
  fi
}

stage_index() {
  say "chunk store and lexical postings (CPU, ~5 min)"
  if done_already artifacts2/lexical/bigram.npz; then
    step "already built; set FORCE=1 to rebuild"
  else
    timed "build chunk store" "$PY" -m legalir2 build-store --corpus corpus.jsonl.gz
    timed "build lexical postings" "$PY" -m legalir2 index-lexical
  fi
  say "lexical baseline on the held-out split - expect Recall 0.8483"
  step "this verifies the corpus arrived intact; anything else means stop"
  "$PY" -m legalir2 predict --no-dense --no-rerank --evaluate --device cpu \
      --output outputs/v2-lexical-validation.json | tail -2
}

stage_train() {
  say "reranker checkpoint"
  if done_already models2/reranker/config.json; then
    step "already downloaded"
  else
    timed "download reranker" "$PY" -m legalir2 download --which reranker
    timed "verify file hashes" "$PY" -m legalir2 verify-models
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

  say "mine training groups from the organisers' labels (CPU, ~3 min)"
  if done_already artifacts2/groups/train.jsonl; then
    step "already mined"
  else
    timed "mine" "$PY" -m legalir2 mine --no-dense --device cpu
  fi
  step "label quality - read this before trusting any loss curve:"
  cat artifacts2/groups/train.stats.json

  say "smoke test the untrained reranker on 20 questions"
  step "cheap insurance: catches a broken model load before the long runs"
  "$PY" -m legalir2 predict --no-dense --limit 20 --evaluate \
      --output outputs/smoke.json | tail -2

  say "fine tune the cross encoder (1-2 h)"
  if done_already models2/reranker-tuned/config.json; then
    step "already trained; set FORCE=1 to retrain"
  else
    step "a step rate prints every 20 steps - check it against the 1-2 h estimate"
    timed "train-ce" "$PY" -m legalir2 train-ce --epochs 2 --group-size 8 --groups-per-step 4
  fi
}

stage_submit() {
  say "measure on the held-out split"
  timed "predict validation" "$PY" -m legalir2 predict --no-dense --evaluate \
      --reranker-model models2/reranker-tuned --output outputs/v2-ce-tuned.json

  say "predict the public test and pack"
  timed "predict public" "$PY" -m legalir2 predict --no-dense \
      --questions public-official.json --reranker-model models2/reranker-tuned \
      --output outputs/submission-v2-ce.json
  "$PY" -m legalir2 validate --predictions outputs/submission-v2-ce.json
  "$PY" -m legalir2 pack --predictions outputs/submission-v2-ce.json \
      --output outputs/submission-v2-ce.zip

  say "done"
  ls -l outputs/submission-v2-ce.zip
  echo "Compare outputs/v2-ce-tuned.metrics.json against the 0.8483 lexical baseline"
  echo "before submitting. If it is not clearly higher, do not submit it."
  stage_results
}

stage_results() {
  say "packaging results for the trip home"
  rm -rf _results && mkdir -p _results/outputs _results/models
  cp outputs/*.json outputs/*.zip outputs/*.log _results/outputs/ 2>/dev/null || true
  for d in models2/reranker-tuned models2/embedding-tuned models2/reranker-distilled; do
    [ -d "$d" ] && cp -r "$d" _results/models/ || true
  done
  cp artifacts2/groups/*.stats.json _results/ 2>/dev/null || true
  du -sh _results
  echo "Pull it down with:  rclone copy $ROOT/_results gdrive:DSC-2026-results -P"
  echo "Trained weights are ~2.3 GB each; delete _results/models for scores only."
}

case "$STAGE" in
  env)     stage_env ;;
  index)   stage_index ;;
  train)   stage_train ;;
  submit)  stage_submit ;;
  results) stage_results ;;
  all)     stage_env; stage_index; stage_train; stage_submit ;;
  *)       echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
