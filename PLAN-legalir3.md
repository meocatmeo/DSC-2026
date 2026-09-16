# legalir3 — stop reranking, start retrieving

legalir2 ends at **0.9493**. Seven interventions were measured after it and every one
landed inside the noise. This document argues that is not bad luck, explains what the
seven had in common, and proposes four changes that do not share it.

---

## 1. The seven failures have one shape

| # | intervention | result |
|---|---|---|
| 1 | multi-positive listwise loss | 0.9330 vs 0.9371 |
| 2 | rerank setting sweep | nothing survived held-out data |
| 3 | cross-fitted stacker refit | −0.0014 [−0.0063, +0.0033] |
| 4 | 1.88B reranker in place of the 0.6B | −0.0038 [−0.0132, +0.0054] |
| 5 | document-level Recall@5 loss, g32 init | −0.0032 [−0.0130, +0.0063] |
| 6 | ViRanker ensemble | +0.0030 [−0.0067, +0.0130] |
| 7 | Qwen3-Reranker ensemble | **+0.0086 [+0.0011, +0.0163]** — but equal to the stacker |

Every one of them **re-scored the same fifty documents.** Different objectives, different
model families, three times the parameters — all of it downstream of a candidate
generator nobody has touched since phase two.

Number 7 is the informative one. It was a real, significant gain over the no-stacker
baseline. Then it measured `+0.0003, 10 better, 10 worse` against the shipped stacker.
Two completely different mechanisms — a second model family, and a learned rank fusion —
converged on the *same* ~0.008 and the same ~20 questions.

That is what a saturated stage looks like. The reranker has extracted what is extractable
from this candidate set, and the eighth way of asking it will find the same 0.008 again.

---

## 2. Where the loss actually sits

```
candidate recall @5 (fusion order)   0.8914     what reranking starts from
candidate recall @50                 0.9805     what reranking can reach
candidate recall @100                0.9845
candidate recall @200                0.9881
final Recall@5                       0.9454
```

Total loss is **0.0546**, in two parts:

| | size | attacked by |
|---|---|---|
| gold never enters the top-50 pool | **0.0195** | nothing, ever |
| gold enters but is not placed top-5 | 0.0351 | all seven interventions |

Reranking converts 0.0540 of the 0.0931 available — **58%**. That conversion rate has
been stable across every variant tried. Treat it as a constant, and the consequence is
immediate:

> Improving the candidate generator pays twice. Gold that enters the pool for the first
> time is worth its full weight, and every point the fusion order improves is carried
> forward at 58%.

Raising `candidate_recall@50` from 0.9805 to 0.99 adds 0.0095 of ceiling outright.
Raising `candidate_recall@5` from 0.8914 to 0.91 adds another ~0.008 through the
conversion. Together that is the step change seven reranker experiments could not find.

---

## 3. The asymmetry nobody noticed

The reranker has been fine-tuned twice, with distribution-matched hard negatives, a
metric-aligned objective, and a learned fusion on top.

**The dense channel ships an off-the-shelf checkpoint.** `submission-v4.run.json` names
`models2/embedding`, whose provenance is `AITeamVN/Vietnamese_Embedding_v2` — the
published weights, not a fine-tune. Half the retrieval system has never been trained on
this corpus at all.

And when it *was* trained, it was trained wrong. This is the important part.

`train_bi.py` draws its negatives from `group["negatives"]`, which `mine.py` fills by
tiering documents on `rank_of` — the **fused** ranking:

```python
tiered = {tier: [] for tier in TIERS}
for doc, rank in rank_of.items():          # rank_of is the FUSED order
```

The single largest win in this project, worth **+0.0210**, was discovering that the
reranker trained on negatives from one distribution and was deployed against another.
The bi-encoder has exactly that bug today. At inference it produces the **dense channel
ranking over the whole corpus, independently**. Its hard negatives should be the
documents *it* ranks highly and gets wrong — not the ones the fusion ranks highly.

`Pipeline.candidates()` already fills `channels["dense"]`. The fix is to tier negatives
on that instead of on `rank_of`. It is perhaps twenty lines.

---

## 4. Four moves, in order

### 4.1 Mine dense-specific negatives and actually ship a tuned embedder

Strongest prior evidence of anything on this list: it is the same fix, on the other model,
and that fix was worth +0.021.

- `mine.py`: tier negatives on `channels["dense"]` when mining for `train-bi`
- `train-bi` on those groups, then rebuild the dense index
- measure `candidate_recall@5` and `@50` — **not** final Recall@5

That last point matters. Judge this stage on candidate recall. Final recall moves at 58%
of any gain, which puts it under the detection floor even when the retrieval improvement
is real and large.

Cost: ~30 min mining, ~2 h training, ~1.2 h re-indexing.

### 4.2 Pseudo-relevance feedback on the lexical channel

Free, classical, and never tried. Run the query, take the top *k* chunks, extract the
terms that discriminate them from the corpus background, re-query with the expanded bag
(RM3). It attacks the vocabulary gap directly: questions are asked in everyday Vietnamese,
statutes are written in formal legal register, and BM25F cannot bridge that on its own.

No model, no parameter budget, no GPU training. Typically +2–4% recall on exactly the
hard queries that are failing here. An afternoon of work.

### 4.3 Late interaction as a third channel

`jinaai/jina-colbert-v2` is on the approved sheet. Late interaction scores every query
token against every document token instead of collapsing both to one vector, which is
worth most precisely where single-vector dense retrieval is weakest: long documents with
one relevant passage — the shape of every document in this corpus.

Budget is not the obstacle (~560M, and the deployment currently uses 1.14B of 4B). The
index is: per-token vectors for the whole corpus. Prototype it on a 2,000-question
subset and measure candidate recall before committing to a full index.

### 4.4 Query expansion with an approved Vietnamese legal LLM

The most upside and the most work. Generate a hypothetical statute passage from the
question and retrieve with that (HyDE), or have the model restate the question in legal
register. Budget permits a 1.7B generator: `Qwen/Qwen3-1.7B`,
`ViLegalQwen3-1.7B-Base`, or `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`, giving
1.7 + 0.57 + 0.57 = **2.84B of 4B**.

Generations cache per question, so the inference cost is paid once for the public test.
Attempt only after 4.1 and 4.2, which are cheaper and better evidenced.

---

## 5. What counts as success

Judge every stage on **candidate recall**, not on final Recall@5, until the retrieval
work is finished. The conversion rate turns a real 0.02 retrieval gain into 0.012 of
final recall — visible — but a real 0.01 into 0.006, which is not.

Targets that would represent a step change rather than another 0.008:

| metric | now | target |
|---|---|---|
| candidate recall @5 | 0.8914 | 0.92 |
| candidate recall @50 | 0.9805 | 0.99 |
| final Recall@5 | 0.9454 | 0.960 |

---

## 6. Carried over from legalir2

Non-negotiable, because each was paid for:

- **Paired tests with intervals, never means.** Four confident conclusions in this
  project turned out to be differences of 0.1–1.4 standard errors.
- **Mining must go through the real candidate generator** — whichever generator the model
  being trained will actually face.
- **Cross-fitting, with folds keyed on normalised question TEXT.** Sixteen duplicate
  question groups exist in the labelled data; ten straddled the folds until this week.
- **The stacker and a second reranker are mutually exclusive** until `build_features`
  reads more than `passage_scores[0]`. Two experiments were wasted on this.
- **Judge a sweep by significance, not by argmax.** Best-of-130 on 400 questions reported
  +0.0107 that did not exist.
- **The leaderboard is not a ruler.** Two submissions agree on ~98% of questions, so a
  public comparison rests on ~20 disagreements — one standard error is about 0.005. Use
  it as a tripwire for gaps above ~0.015, never as a selector.
