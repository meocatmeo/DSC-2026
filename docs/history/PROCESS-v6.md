# UIT DSC 2026 — Task 1 (LegalIR): the whole project, and where the ceiling is

Public test: **0.9493**. Ten measured interventions followed it. One was statistically
real and turned out to be redundant; nine were not distinguishable from no change.

This document records what was built, what was measured, and — the part worth reading —
the mechanism behind the ceiling, which the last experiment finally exposed.

*Supersedes the earlier draft of this file, which described the wrong architecture in
section 1 and drew the wrong conclusion in section 7. Both are corrected below.*

---

## 1. The system, as actually shipped

Verified from `submission-v4.run.json`, not from memory:

```
question
   |
   +-- lexical channel   BM25F + hashed bigram postings, article-level chunks
   +-- dense channel     AITeamVN/Vietnamese_Embedding_v2, PUBLISHED weights
   |
   v
weighted RRF fusion  ->  top-50 candidates
   |
   v
cross-encoder rerank  ->  AITeamVN/Vietnamese_Reranker fine-tuned (reranker-g32),
                          sequence-classification head, 3 passages per document
   |
   v
learned rank fusion (stacker, 12 features)  ->  final top-5
```

Deployed parameters: 567,754,752 embedder + 567,755,777 reranker = **1.14B of a 4B
allowance**. Rerank depth is **50**, not 100. Every checkpoint is from the organisers'
approved sheet; `download` is the only command that touches the network and it records
a SHA-256 for every file it writes.

Recomputing the saved validation predictions gives **0.945397**, matching the stored
metric exactly.

---

## 2. Score history

| version | change | validation | public |
|---|---|---|---|
| v1 | lexical only | 0.8483 | — |
| v2 | + dense channel, + fine-tuned reranker | — | 0.9223 |
| v3 | **hard negatives mined through the real candidate generator** | 0.9371 | 0.9433 |
| v4 | **+ learned rank fusion over 12 features** | 0.9454 | **0.9493** |
| v5–v8 | ten further interventions | 0.9276–0.9457 | not shipped |

Two changes produced everything.

---

## 3. What worked, and why

### 3.1 Mining through the real candidate generator — **+0.0210**

`mine.py` had a private `_blend()`: min-max normalisation, equal channel weights.
Inference used weighted RRF. Two reasonable implementations of the same idea, in two
files, producing two different candidate orderings.

The reranker was therefore trained to reject negatives from distribution A and deployed
against distribution B. Nothing errors; the model just quietly converts fewer of the
candidates it is handed.

The fix was to delete the second implementation and mine through `Pipeline.candidates()`
with the reranker unloaded. A regression test now asserts mined negatives are a subset of
the prediction ranking under deliberately lopsided channel weights — a configuration
where the two implementations would have disagreed loudly.

Larger than every subsequent intervention combined.

### 3.2 Learned rank fusion — **+0.0060**

Five slots were being chosen from a single fused score. A document ranked first by BM25F
and fortieth by the dense channel is a different object from one ranked twentieth by
both, and fusion destroys that distinction before anything can use it.

`stack.py` fits a linear listwise model over 12 per-question min-max-normalised
features — reranker max/second/mean/spread/near-top, fused score and rank, lexical score
and rank, dense score and rank, lexical-dense gap — with multi-positive targets.

---

## 4. How results are judged

Partway through, a set of confident conclusions turned out to be noise: three
"regressions caused by improving retrieval" were differences of 0.1 to 1.4 standard
errors on a 1,050-question holdout where 1 s.e. is 0.0075.

Everything since is judged by `compare.py`: paired per-question scores, a 95% bootstrap
interval, an exact sign test, and the count of questions that actually changed.

Four rules earned the hard way:

**A sweep is a selection procedure, not a measurement.** Best-of-130 configurations on
400 questions reported +0.0107; none survived held-out data. `cache.search()` now
requires a paired interval excluding zero before adopting anything.

**Scoring a model on its own training data measures memorisation.** The frozen-score
cache is built from cross-fitted models — questions split in two, a reranker trained on
each half, every question scored by the model that never saw it.

**Folds must key on normalised question TEXT.** The labelled set contains 16 duplicate
question groups; 10 straddled the fold boundary while folds were keyed on id, leaking
training questions into the held-out half of the cross-fit.

**The detection floor is not a constant.** It depends on the distribution of paired
differences, not only on *n*. In practice, for changes that produce wins and losses in
roughly equal numbers — which every intervention here did — it sits near 0.0094 on 1,050
questions.

---

## 5. The ten interventions

| # | intervention | result |
|---|---|---|
| 1 | multi-positive listwise loss | 0.9330 vs 0.9371 |
| 2 | rerank setting sweep | nothing survived held-out data |
| 3 | cross-fitted stacker refit | −0.0014 [−0.0063, +0.0033] |
| 4 | 1.88B reranker (Prism-Qwen3.5) in place of the 0.6B | −0.0038 [−0.0132, +0.0054] |
| 5 | document-level Recall@5 loss, g32 init | −0.0032 [−0.0130, +0.0063] |
| 6 | document-level Recall@5 loss, base init, scale-invariant | −0.0031 |
| 7 | ViRanker ensemble | +0.0030 [−0.0067, +0.0130] |
| 8 | **Qwen3-Reranker-0.6B ensemble** | **+0.0086 [+0.0011, +0.0163]** |
| 9 | tuned embedder + tuned reranker + stacker | −0.0006 [−0.0090, +0.0078] |
| 10 | retrain on all 7,000 labels | unmeasurable by construction |

Number 8 was real — and then measured `+0.0003, 10 better, 10 worse` against the
stacker. Two unrelated mechanisms, a second model family and a learned rank fusion,
converged on the same 0.008 and the same ~20 questions.

Notes on two of the failures, because the *reason* matters more than the result:

**#4 and #8 were partly confounded.** `final_order` short-circuits into the stacker
whenever one is active, and `build_features` reads `passage_scores[0]` only — so a
second reranker is scored, paid for, and then ignored. Both experiments measured a new
model filtered through a fusion fitted on the old one's score distribution.

**#5 had no gradient.** `softplus(s_negative − s_gold)` is scale dependent, and a
competent reranker separates gold from mined negatives by 10–16 raw logits.
softplus(−16) is 1.1e-7. The loss was standardised per question in #6 and the answer did
not change, which is what closes the hypothesis rather than the implementation.

---

## 6. Where the loss sits

```
candidate recall @5 (fusion order)   0.8914     what reranking starts from
candidate recall @50                 0.9805     what reranking can reach
candidate recall @100                0.9845
final Recall@5                       0.9454
```

Total loss **0.0546**: 0.0195 never enters the top-50 pool, 0.0351 enters and is not
placed top-5. Concentrated by question type:

| | questions | Recall@5 | share of loss | recoverable by reordering the actual 50 |
|---|---|---|---|---|
| single-gold | 957 | 0.9613 | 65% | 0.0257 |
| multi-gold | 93 | 0.7814 | 35% | 0.0094 |

Multi-gold questions are 8.9% of the set and over a third of the loss, but most of their
loss is gold that never reached the pool — so the larger *recoverable* mass is in
single-gold questions.

---

## 7. The ceiling, and what causes it

The earlier draft of this section proposed that retrieval was the untouched half and
would pay twice. That was wrong, and experiment #9 is why.

A tuned bi-encoder and a matching dense index were built in phase two. Three runs used
them:

```
                          cand@5    @50    final   conversion
base + stacker            0.8914  0.9805  0.9454     61%
tuned + stacker           0.9171  0.9833  0.9448     42%
tuned, no stacker         0.9171  0.9833  0.9333     25%
tuned, no stacker (v2)    0.9171  0.9833  0.9276     16%
```

The tuned embedder is a **large** retrieval win: +0.0257 of candidate recall@5, three
times anything the reranker work produced, plus a higher ceiling at every depth.

And the finished system does not move. 0.9448 against 0.9454, with **1,019 of 1,050
predictions identical**.

**Retrieval and reranking are substitutes here, not complements.** The gold a better
dense channel lifts into the top five is gold the reranker was already recovering from
deeper in the pool. Improving retrieval hands the reranker work it was already doing,
and the conversion rate falls by almost exactly what retrieval gained.

That explains the whole sequence. Ten interventions and two retrieval configurations all
land at 0.945 ± 0.006 because the binding constraint is neither stage. It is the ~0.035
of gold that reaches the pool and cannot be placed in the top five by *any* ordering of
the same evidence.

The error inventory says what that gold is:

- query `123452`: the amendment `226545` was candidate 5 and dropped; the base decree
  `258342` was kept
- query `132594`: gold `303485` was candidate 8; the output was five related
  tax-extension documents
- query `9060`: the sole gold was candidate 14 and never appears as a positive anywhere
  in the 5,950 training questions

No reranker separates a decree from the decree amending it on text alone. That is what
"amendment" means. The residual is not a ranking failure — it is a *distinguishability*
failure, and it is the subject of `PLAN-legalir4.md`.

---

## 8. Reproducing

Two interpreters, deliberately:

| | transformers | used for |
|---|---|---|
| `.venv` | 4.51.3 | the shipped pipeline |
| `.venv5` | 5.17.0 + flash-linear-attention | Qwen3.5 checkpoints only |

Exact package sets in `outputs/venv-freeze.txt` and `outputs/venv5-freeze.txt`.
`gpu-restore.sh` rebuilds a box from Drive; its lexical-only baseline must read
**0.8483**, which is the corpus integrity check.

Hardware: the full sequence runs on a 24 GB card. Rent by max duration before price.
