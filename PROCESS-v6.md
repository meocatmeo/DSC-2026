# UIT DSC 2026 — Task 1 (LegalIR): what was done, what worked, and what the numbers say

**Active configuration note:** the Qwen 2B phase described later is historical and
unshipped. The active improvement path stays on the 0.6B AITeamVN reranker that
produced the 0.9493 submission; use `gpu-phase4.sh multipos` with `--document-loss`.

Public test: **0.9493**. This document explains how it got there, what was tried after and
failed, and — the part worth reading — why the failures are informative rather than just
disappointing.

---

## 1. The system

A three-stage retrieval pipeline over a Vietnamese legal corpus, answering each question
with five document ids, scored by Recall@5.

```
question
   |
   +-- lexical channel   BM25F + bigram postings over a chunk store
   +-- dense channel     Qwen3-Embedding-0.6B, fine-tuned, FAISS-free resident matrix
   |
   v
weighted RRF fusion  ->  top-100 candidates
   |
   v
cross-encoder rerank  ->  Qwen3-Reranker-0.6B, fine-tuned (causal yes/no scoring)
   |
   v
learned rank fusion (stacker)  ->  final top-5
```

Deployed parameter budget: 0.57B embedder + 0.6B reranker = **1.2B of a 4B allowance**.
Every checkpoint comes from the organisers' approved model sheet. No external API is
called at any point; `download` is the only command that touches the network, and it
records a SHA-256 for every file it writes.

---

## 2. Score history

| version | change | public |
|---|---|---|
| v1 | lexical only | 0.8483 |
| v2 | + dense channel, + fine-tuned 0.6B reranker | 0.9223 |
| v3 | **hard negatives mined through the real candidate generator** | 0.9433 |
| v4 | **+ learned rank fusion over 12 per-question features** | **0.9493** |
| v5 | cross-fitted re-fit of the stacker | −0.0014, not shipped |
| v6 | 1.88B reranker in place of the 0.6B | −0.0038, not shipped |

Two changes produced everything. Both are described below, because *why* they worked is
more transferable than the fact that they did.

---

## 3. The change that mattered most: mining through the real candidate generator

Hard-negative mining originally used a private `_blend()` inside `mine.py` — min-max
normalisation, equal channel weights. Inference used weighted RRF in `pipeline.py`. Two
different functions, both reasonable, producing two different candidate orderings.

The consequence is subtle and expensive. The reranker was trained to separate gold from
negatives drawn from distribution A, and then deployed against negatives drawn from
distribution B. It had never seen the documents it would actually be asked to rank.

The fix was to delete the second implementation and mine through `Pipeline.candidates()`
with the reranker unloaded, so training negatives are by construction the same documents
inference will produce.

**+0.0210 on the public test.** Larger than every subsequent intervention combined.

A regression test now asserts that mined negatives are a subset of the prediction
ranking under deliberately lopsided channel weights — a configuration where the two
implementations would have disagreed loudly.

---

## 4. The second change: learned rank fusion

Recall@5 is decided by five slots, but the pipeline was choosing them from a single
fused score. A document ranked first by BM25F and fortieth by the dense channel is a
different object from one ranked twentieth by both, and fusion had already destroyed
that distinction before anything could use it.

`stack.py` fits a linear listwise model over 12 per-question min-max-normalised
features — reranker score max/second/mean/spread/near-top, fused score and reciprocal
rank, lexical score and rank, dense score and rank, and the lexical-dense gap — with a
softmax cross-entropy objective whose target mass splits across multiple gold documents.

**+0.0060 on the public test.** The only one of four phase-four interventions to survive
a paired test.

---

## 5. How results are judged, and why that changed mid-project

Partway through, a set of confident conclusions turned out to be noise. Three "regressions
caused by improving retrieval" were differences of 0.1 to 1.4 standard errors on a
1,050-question holdout, where 1 s.e. is 0.0075. They were nothing, read as something.

Everything since is judged by `compare.py`: a **paired** comparison over per-question
scores, reporting the mean difference, a 95% bootstrap interval, an exact sign test, and
the count of questions that actually changed.

Three consequences worth stating plainly:

**The detection floor is 0.0094 at n=1050.** Any change smaller than that is
unmeasurable on the holdout, whether or not it is real.

**Coordinate-ascent sweeps are a selection procedure, not a measurement.** A phase-four
sweep took the argmax of ~130 configurations on 400 questions and reported +0.0107. None
of it survived held-out data. `cache.search()` now requires a candidate to beat the
incumbent on a paired interval that excludes zero before it is adopted — conservative by
construction, and it will leave small real gains on the table rather than hand back
gains that are not there.

**Scoring with a model on its own training data measures memorisation.** The deployed
reranker saw 5,950 of the ~7,000 labelled questions. The frozen-score cache is therefore
built from *cross-fitted* models — the questions split in two, a reranker trained on each
half, every question scored by the model that never saw it. Without that, a larger
evaluation set is just a more confident way to be wrong.

The cache itself makes the expensive part reusable: the reranker's forward pass does not
depend on any ranking-level setting, so scores are computed once and every question about
aggregation, depth, blend weights or the stacker becomes arithmetic. Resolution across
all ~7,000 questions is 0.0036.

---

## 6. Phase six: the 1.88B reranker

**The hypothesis.** Cross-encoder quality scales with parameter count, and the deployment
used 1.2B of a 4B allowance. This was the one remaining change that adds capacity rather
than re-weighting what is already there — better prior evidence than anything else left
on the list.

**The model.** `infgrad/Prism-Qwen3.5-Reranker-2B` (approved sheet), 1.88B parameters,
24 layers, hidden 2048. Deployed budget would have been 2.45B of 4B.

### What it took to run at all

*Architecture.* `model_type: qwen3_5_text` is unknown to transformers 4.51.3 — a hard
`KeyError`. Qwen3.5 needs 5.2+; current release is 5.17.0. That is a major-version jump
for every tokeniser and model in the project.

Rather than upgrade in place, `gpu-venv5.sh` builds a **second interpreter**. The risk of
upgrading was never that it would crash — a crash announces itself. The risk was that
everything would load and something small would shift, so that phase six compared a new
model under a new library against a baseline measured under the old one and reported the
sum of two changes as one. The second venv also re-measures the shipped 0.6B pipeline
under transformers 5, giving a baseline where the library variable cancels.

*Prompt template.* Prism's system prompt and instruction differ from Qwen3-Reranker's.
Using the wrong one raises nothing at all — it just makes every score quietly worse, and
the symptom four hours later is "the 2B didn't help." `models.py` now carries a
per-checkpoint template resolved through provenance, and the preflight asserts that a
known-relevant passage outscores a known-irrelevant one before any training begins.
It passed at +0.7549 against +0.1480.

*Kernels.* Qwen3.5 is a hybrid linear-attention architecture. Both fast paths
(`chunk_gated_delta_rule`, `causal_conv1d_fn`) were missing, so it ran the reference
implementation at 52 passages/s.

Installing `flash-linear-attention` appeared to make it **seven times slower** — 7 p/s.
That was the benchmark, not the kernel: FLA JIT-compiles Triton kernels per tensor shape
on first call, and the probe timed one un-warmed batch. A rewritten benchmark that warms
up at each shape and reports four repetitions showed steady state at **68–77 p/s**.

`causal-conv1d` was abandoned mid-build. Its `setup.py` hardcodes a `-gencode` list
rather than reading `TORCH_CUDA_ARCH_LIST`, so it was compiling `sm_100` kernels for an
`sm_89` card — thousands of template instantiations, six architectures of which were
useless, for a few percent on the smaller of the two kernels.

*Memory.* Gradient checkpointing off OOMed at 47.37 GiB on a 48 GB card. The estimate
that said 22–28 GB failed to account for per-chunk recurrent state in the hybrid layers,
the 150k-vocab embedding, and autocast's bf16 weight copies.

### Training

LoRA r=16 on all attention and MLP projections, merged on save so the deployed artefact
is an ordinary checkpoint with no peft dependency. Group size 16, accumulate 8 (128 pairs
per optimiser step), 2 epochs, max length 768, on the **already-mined** groups — the
candidate distribution had not changed, so those negatives were still the ones a 2B model
would face at inference, and re-mining would only have risked the section-3 property.

1.93 s per forward, 11,900 forwards, **6.4 hours**. Roughly 96 TFLOPS sustained — close
to what the card can deliver, so compute-bound rather than misconfigured.

One note on how *not* to shorten it: the cosine schedule is sized from the epoch count,
so `--epochs 1` is a complete run that warms up and anneals to zero. Killing a two-epoch
run halfway is not the same thing — it leaves the learning rate near its peak and saves a
mid-training snapshot. (That exact bug, sizing the schedule in micro-steps rather than
optimiser steps, cost a training run earlier in the project. Nothing in the loss curve
shows it.)

### Result

```
2B reranker vs the 0.6B, same library, paired, n=1050

  before            0.9454
  after             0.9416
  difference       -0.0038   95% CI [-0.0132, +0.0054]
  disagreements    17 better, 22 worse, 1011 identical
  sign test         p = 0.522

  VERDICT: not distinguishable from no change
```

Not shipped.

### The number that matters more than the verdict

**1,011 of 1,050 rankings were identical.** Tripling reranker capacity moved 3.7% of the
output. That is not a model that failed to help; it is a model that barely got a vote.

There is a specific reason, and it is a design error in the harness. `final_order`
short-circuits into the stacker whenever one is active, and `v4-stacker.json` was fit on
features drawn from the **0.6B's** score distribution — `rr_max`, `rr_second`,
`rr_spread`, `rr_near_top`. Substituting a model whose scores live on a different scale
does not retrain those weights; it feeds new numbers into a fusion that learned what the
old ones meant.

So this measured "a 2B reranker filtered through a stacker that does not know it," not
"a 2B reranker." The experiment is confounded.

Refitting properly needs 2B features over the training questions — 5,950 × 100 documents
× 3 passages at ~50 p/s is about ten hours before anything is measured. On a path where
five consecutive interventions have landed inside the noise, that was judged a bad bet.

---

## 7. Where the remaining loss actually is

From `outputs/v6-2b.metrics.json`:

```
candidate pool @5      0.8914     fusion order, before reranking
candidate pool @20     0.9679
candidate pool @100    0.9845     ceiling any reranker can reach at this depth
candidate pool @200    0.9881
final Recall@5         0.9454
```

Reranking converts 0.0540 of the 0.0931 available — **58% of the headroom**. What remains
is **0.0391** of gold that reaches the candidate pool and still does not land in the top
five.

Retrieval is not the ceiling. Only 1.55% of gold never enters the pool at all, and
doubling depth to 200 would add 0.0036 to the ceiling. The leak is in ranking.

And it is concentrated:

| | questions | Recall@5 | share of total loss |
|---|---|---|---|
| single-gold | 957 | 0.9613 | 65% |
| multi-gold | 93 | 0.7814 | **35%** |

Multi-gold questions are 8.9% of the set and over a third of everything being lost. The
mechanism is structural: Recall@5 on a 2-gold question requires *both* documents in five
slots, so placing one of them scores 0.5 regardless of the reranker's confidence. Nothing
tried so far addresses this — top-5 selection treats a 2-gold question exactly like a
1-gold one.

---

## 8. Why five interventions in a row measured as nothing

Not a failure of ideas. It is the regime.

Detection floor on the holdout is 0.0094. Clearing it takes a single change that converts
**26% of all remaining ranking headroom at once**. Every plausible remaining fix —
multi-gold-aware selection, a refit stacker, deeper reranking — is individually worth
somewhere in the 0.003–0.008 band. They are real and they are invisible to the ruler that
can be afforded.

The cross-fitted 7,000-question harness resolves 0.0036 and exists and works. It costs
GPU hours per evaluation, and when it was run it found nothing either — including
measuring the cross-fitted stacker at −0.0014 [−0.0063, +0.0033] against the shipped one,
which is itself reassuring evidence that 0.9493 is not overfit.

---

## 9. If this is picked up again

The only target with enough mass behind it to be measurable is **multi-gold selection**:
20.3 of 57.4 lost points inside 93 questions, and the actual mechanism is untouched.

Everything else is below the floor individually. Stacking several such changes and
measuring the bundle would work, but a bundle that measures positive does not say which
part earned it — and on this project, that has mattered.

### Reproducing the environment

Two interpreters, deliberately:

| | transformers | used for |
|---|---|---|
| `.venv` | 4.51.3 | the shipped 0.9493 pipeline |
| `.venv5` | 5.17.0 + `flash-linear-attention` | anything Qwen3.5 |

Exact package sets are in `outputs/venv-freeze.txt` and `outputs/venv5-freeze.txt`.

`gpu-restore.sh` rebuilds a box from Drive and prints what survived. Its lexical-only
baseline must read **0.8483** — that is the corpus integrity check, and a different
number means every later measurement is against a different dataset.

### Hardware

1x RTX 6000 Ada, 48 GB, ~764 GB/s measured, $0.881/h. Phase six end to end was about
ten hours and roughly $9. Rent by **max duration** before price: the full sequence does
not fit inside a three-hour ceiling.
