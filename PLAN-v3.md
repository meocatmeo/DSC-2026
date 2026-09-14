# Phase 3 plan: 0.9433 → 0.96

**Third revision, now grounded in the actual diagnostics** rather than in guesses from
the old system. Two earlier drafts are superseded: the 2B reranker (anchoring on unused
parameter budget) and the assumption that raising candidate recall helps. The data says
otherwise, clearly.

Target GPU: **24 GB**. Nothing built yet.

---

## 1. What the diagnostics say

Validation, 1050 questions. `v2-g32` is the run that produced 0.9433 on the public test.

| run | R@5 | cand@50 | cand@100 | conversion |
|---|---|---|---|---|
| v2-ce-tuned (lexical only) | 0.9170 | 0.9608 | 0.9752 | 95.4% |
| v2-dense-ce (+dense, old CE) | 0.9243 | 0.9805 | 0.9845 | 94.3% |
| **v2-g32 (matched negatives)** | **0.9371** | 0.9805 | 0.9845 | **95.6%** |
| v2-ens (+second reranker) | 0.9365 | 0.9805 | 0.9845 | 95.5% |
| v2-bi-tuned (tuned embedder) | 0.9333 | 0.9833 | **0.9903** | 94.9% |
| v2-deep (candidates=100, passages=4) | 0.9276 | 0.9833 | 0.9903 | 94.3% |

### Finding 1 — every attempt to give the reranker *more* made it worse

- The **tuned bi-encoder** produced the best ceiling of the whole project, candidate
  recall@100 of 0.9903, and the final score **fell** 0.9371 → 0.9333.
- Showing the reranker **100 candidates instead of 50** dropped it further, to 0.9276.
- Adding a **second reranker** dropped it slightly, 0.9371 → 0.9365.

Three independent interventions, three regressions. **Stop chasing candidate recall.**
The pool is not the constraint and enlarging it actively hurts, because every extra
candidate is another chance to promote a wrong document into five fixed slots.

This also retrospectively justifies the autopilot's provenance-based selection: it
picked `v2-g32` over the runs with better retrieval, which was correct.

### Finding 2 — the loss is a ranking problem, not a retrieval problem

97 gold documents are missed. **77 of them (79%) were retrieved, scored by the reranker,
and then ranked below fifth.** Only 20 were never retrieved at all.

### Finding 3 — the loss is concentrated in multi-gold questions

| bucket | questions | mean recall | recall mass lost | share of the gap |
|---|---|---|---|---|
| single-gold (\|gold\| = 1) | 957 | 0.9551 | 0.0409 | 65% |
| multi-gold (\|gold\| = 2) | 78 | 0.8013 | 0.0148 | 23% |
| multi-gold (\|gold\| ≥ 3) | 15 | 0.5000 | 0.0071 | 11% |
| **total** | **1050** | **0.9371** | **0.0629** | **100%** |

**8.9% of questions produce 34% of the lost recall mass and 56% of all missed gold
documents.** On questions with three or more gold documents the system finds *half*.

Of the 54 gold documents missed on multi-gold questions, **41 (76%) are sitting in the
pool**, already scored, simply not promoted.

### Finding 4 — the reranker overrides retrieval, and is usually right

| | count |
|---|---|
| retrieval had it in top-5, reranker kept it | 982 |
| retrieval had it in top-5, **reranker dropped it** | **23** |
| retrieval missed it, reranker rescued it | 80 |
| neither | 74 |

Net effect of reranking: **+57 gold documents**. The reranker earns its place. But 23
documents were discarded that retrieval had already ranked correctly, and those are
recoverable by weighting the two signals properly instead of at 1.0 versus 0.02.

---

## 2. The revised plan

### Move B (promoted to first) — multi-positive loss

The largest concentrated, well-understood pool of recoverable points.

`train_ce.py` picks **one** positive per group (`rng.choice(group["positives"])`) and
optimises softmax cross-entropy. For a question with three gold documents, the model is
trained to rank *one* of them first and is never told the other two also belong in the
top five. The metric, meanwhile, gives partial credit for each.

The fix is to spread the target mass across every gold document in the group — a
multi-positive listwise loss. It reduces exactly to the current loss when a group has
one positive, so single-gold behaviour is unchanged by construction.

- **Cost: one training run, identical to today's.** No new model, no extra VRAM.
- **Available: +0.0179** if the gap closed completely; realistically **+0.007 – 0.012**.
- **Evidence: strong.** 41 of the 54 missed documents are already in the pool and
  already scored.

### Move A — learned rank fusion

Replace `final = reranker × 1.0 + fusion_prior × 0.02` (a constant I typed, never
fitted) with a model trained on the organiser's labels over features that already exist:
reranker score statistics, BM25F score and rank, bigram contribution, dense score and
rank, RRF rank, plus structural features such as *does the query contain this document's
number*.

- **Cost: CPU, minutes.** 5,950 × ~50 ≈ 300k rows, ~15 features. No GPU, nothing
  against the 4B budget.
- **Target: the 23 documents** the reranker discarded that retrieval had ranked
  correctly, without giving back too many of the 80 it rescued.
- **Expected: +0.005 – 0.010.**
- **Rules:** fits weights on the organiser's labels exactly as the BM25F and fusion
  weights already are. It is a learned version of the `rerank_blend` constant.

### Move C — sweep `rerank_blend` and `context_radius` (free)

One GPU pass, no training. `context_radius: 0` means the reranker judges a 320-word
chunk in isolation while accepting 2304 tokens. Note from Finding 1: the sweep should
**not** be expected to favour `candidates=100` — `v2-deep` already measured that as a
regression. Let it confirm or overturn that on its own evidence.

### Dropped, on the evidence above

- **Raising candidate recall** — RM3, a third retrieval channel, the tuned bi-encoder.
  Measured as counterproductive three separate ways. The tuned embedder can only be used
  if the reranker's negatives are re-mined through *it*, which is another full cycle.
- **The 2B reranker** — still the biggest single expected gain, but the cheap moves now
  have measured evidence behind them and it does not. Revisit only if A+B fall short.
  It also carries a permanent 3× inference tax and a possible latency-rule problem.
- **A second reranker for the ensemble** — `v2-ens` already measured this as neutral to
  negative with two checkpoints from the same base.

---

## 3. Sequencing

| # | stage | cost | expected |
|---|---|---|---|
| 1 | multi-positive loss, retrain, measure | ~2.5 h GPU | +0.007 – 0.012 |
| 2 | learned rank fusion | minutes CPU + 0.3 h GPU to measure | +0.005 – 0.010 |
| 3 | sweep `rerank_blend`, `context_radius` | ~0.7 h GPU | +0.002 – 0.005 |
| 4 | pack and submit best by validation | ~0.5 h GPU | |
| | **total** | **~4 GPU hours** | |

Validation ran ~0.006 below public on this system (0.9371 → 0.9433), so a validation
score of ~0.952 should correspond to roughly 0.958 public.

---

## 4. Honest expectation

| | validation | implied public |
|---|---|---|
| now | 0.9371 | 0.9433 |
| + moves A, B, C | 0.950 – 0.960 | 0.956 – 0.966 |
| hard ceiling at candidate@50 | 0.9805 | — |

**0.96 is now more likely than not**, from ~4 GPU hours and no new models — a better
position than either earlier draft, and it rests on measurements rather than on
literature estimates.

The residual after that is genuinely hard: 20 documents never retrieved, plus
single-gold questions where the reranker simply prefers a wrong document. Some of that
is label noise — Vietnamese law republishes consolidated and amending versions, so a
"wrong" answer is sometimes a near-identical restatement the labels do not list.

---

## 5. Code to be written (scope, for approval)

- `train_ce.py`: multi-positive listwise loss; must reduce to the current loss when
  `len(positives) == 1`, with a test asserting exactly that.
- `mine.py`: emit per-gold-document positives so the loss can address them separately
  (currently positives from all gold documents are pooled into one flat list).
- New `legalir2/stack.py`: feature extraction over cached candidates, fit, and a
  `final_order`-compatible scorer so the live pipeline and any tuner share one path —
  the same discipline that `final_order()` already enforces.
- `tune.py`: `context_radius` in `RERANK_GRID`.
- `gpu-phase4.sh` / `gpu-auto4.sh`: reuse the shipping, backup, provenance-selection and
  budget-audit machinery that survived the 7-hour run.
- Tests: multi-positive reduces to current on single-positive groups; the stacker's
  offline ranking reproduces the live pipeline's order exactly.
