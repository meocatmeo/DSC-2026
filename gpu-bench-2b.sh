#!/usr/bin/env bash
# Steady-state throughput for the 2B reranker. Separates compilation from computation.
#
#   bash gpu-bench-2b.sh
#
# WHY THIS EXISTS
# ---------------
# The preflight scored 64 passages once and divided by the wall clock. That is a fine
# sanity check and a terrible benchmark, and installing flash-linear-attention proved it:
# throughput "fell" from 52 to 7 passages/s, which would mean the optimised kernel is
# seven times slower than the reference one. What actually happened is that FLA compiles
# its Triton kernels on first call, per tensor shape, and the probe timed the compiler.
#
# So: warm up at the exact shape being measured, discard that, then time several
# repetitions and report each one. If repetition 1 is slow and 2-4 are fast, the cost was
# compilation - paid once per process, irrelevant across a 3-hour run. If all four are
# equally slow, the kernel really is worse here and should come back out.
#
# Batch size matters for the same reason: linear-attention kernels are written for long
# sequences and fat batches, and a 2-passage call has nothing to amortise over. The sweep
# below is what tells you which RERANK_BATCH to actually pass to phase six.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv5/bin/python}"
MODEL="${MODEL:-models2/reranker2b}"
BATCHES="${BATCHES:-8 16 32 64}"
REPEATS="${REPEATS:-4}"
MAXLEN="${MAXLEN:-768}"

printf '\n=== which kernels are installed ===\n'
"$PY" - <<'KERNELS'
import importlib.util as u
for name, pip in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
    found = u.find_spec(name) is not None
    print(f"  {pip:<24} {'installed' if found else 'NOT installed'}")
KERNELS

printf '\n=== steady-state throughput ===\n'
"$PY" - "$MODEL" "$BATCHES" "$REPEATS" "$MAXLEN" <<'BENCH'
import sys, time
import torch
from legalir2.rerank import Reranker
from legalir2.store import Store

path, batches, repeats, maxlen = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4])

store = Store("artifacts2/store.sqlite")
gold = store.doc_ids[0]
text = store.chunk_text(store.document_chunk_ids(gold)[0], with_label=False)
query = " ".join(text.split()[:15])
store.close()

model = Reranker(path, device="cuda", max_length=maxlen)

print(f"  {'batch':>6}  {'run 1':>9}  {'run 2':>9}  {'run 3':>9}  {'run 4':>9}   {'steady':>9}")
print(f"  {'':>6}  {'(warm)':>9}  {'':>9}  {'':>9}  {'':>9}   {'p/s':>9}")
best = (0.0, None)
for size in (int(b) for b in batches):
    passages = [text] * size
    # Warm up at THIS shape. Triton caches per shape, so a warmup at batch 8 does
    # nothing for batch 32 - which is exactly how the first benchmark went wrong.
    model.score(query, passages, batch_size=size, max_length=maxlen)
    torch.cuda.synchronize()

    rates = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.score(query, passages, batch_size=size, max_length=maxlen)
        torch.cuda.synchronize()
        rates.append(size / (time.perf_counter() - started))
    steady = sum(rates[1:]) / max(1, len(rates) - 1)   # drop run 1 even post-warmup
    cells = "  ".join(f"{r:9.1f}" for r in rates)
    print(f"  {size:>6}  {cells}   {steady:9.1f}", flush=True)
    if steady > best[0]:
        best = (steady, size)

rate, size = best
print(f"\n  best: {rate:.0f} passages/s at batch {size}")
print(f"  1,050-question validation at 100 docs x 3 passages: "
      f"{1050 * 300 / rate / 3600:.1f} h")
print(f"  a 2-epoch LoRA pass over 5,950 groups of 16 (fwd+bwd ~3x): "
      f"{5950 * 2 * 16 * 3 / rate / 3600:.1f} h")
model.close()
BENCH

cat <<'READ'

=== how to read this ===
  run 1 slow, runs 2-4 fast   -> that was compilation. Paid once per process, so it
                                 vanishes across a three-hour run. Keep the kernel.
  all four equally slow       -> the kernel is genuinely worse at this shape:
                                 .venv5/bin/pip uninstall -y flash-linear-attention
  throughput rises with batch -> pass the winning size as RERANK_BATCH to phase six.

  Compare the best number against the 52 p/s the reference implementation managed at
  batch 32. That is the only comparison that decides anything.
READ
