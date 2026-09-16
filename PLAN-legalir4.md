> **TESTED AND NOT SUPPORTED.** The three diagnostics in §2 were run on the 1,050
> validation questions against the corpus metadata. Results, and what they mean, are in
> §0 below. The proposal in §3 is not worth building. Kept because the diagnostics
> themselves produced the most useful map of the residual this project has.

# 0. Verdict — the diagnostics say no

Parsing the corpus recovers a document number for 81.6% of the 8,532 documents, a type
for 83.6%, and an amendment marker for 5.4%. So the metadata is there. It does not help.

**D1 — citations. Dead outright.**

| | |
|---|---|
| questions containing a document number | **5 of 1,050 (0.5%)** |
| ... where a gold document's number matches | 5 (100%) |
| `cite_exact` fires on | 11 (question, candidate) pairs |
| precision when it fires | 54.5% |
| questions containing a year | 43 (4.1%) |
| ... where a gold document's year matches | 9 (20.9%) |

The feature is perfectly precise on the five questions that cite a document and
irrelevant everywhere else. People asking these questions describe a situation; they do
not cite the statute they are looking for. That was an assumption, and it was wrong.

**D2 — sibling confusion. Real but small, and far weaker than the anecdotes suggested.**

| | failures | control |
|---|---|---|
| at least one predicted document related to the missed gold | 47.4% | **42.1%** |
| pairwise (missed gold, predicted) related | 17.7% | 9.8% |

The pairwise lift is 1.8x, which is a genuine signal. But the measure that matters for a
top-5 metric — does a sibling appear at all — is 47.4% against a 42.1% control. Siblings
crowd the top five of *most* questions, including the ones the system gets right.

The three examples in `ANALYSIS-v7` are real, and they are not representative of 76
failures. Generalising from them was a mistake this diagnostic caught for the price of an
afternoon, which is exactly what it was for.

**D3 — multi-gold relatedness. 2.5x lift on a tiny base.**

15.1% of the 126 multi-gold pairs are structurally related against a 6.15% random
control — so relatedness *is* evidence. But that is 19 pairs across 93 questions. Even
perfect co-selection there moves the overall metric by well under the detection floor.

# 0.1 What the diagnostics did produce

The most useful map of the residual so far. Of the 86 gold documents missing from the
final five:

| candidate rank | count | recall lost | share |
|---|---|---|---|
| already in candidates 1–5 | 14 | 0.0089 | 16% |
| 6–10 | 10 | 0.0071 | 13% |
| 11–20 | 15 | 0.0113 | 21% |
| 21–50 | 11 | 0.0078 | 14% |
| 51+ (beyond rerank depth) | 16 | 0.0076 | 14% |
| never retrieved at all | 20 | 0.0119 | 22% |

Three things follow.

**The loss is diffuse.** No bucket holds more than 22%, and no structural pattern
explains any bucket. There is no lever here, which is why ten interventions each moved
about the same 0.008 and no more.

**0.0089 is actively destroyed.** Fourteen gold documents were in the fusion's top five
and the ranking stage pushed them out. That is 16% of the loss, and it is the same
phenomenon as experiment #9: the reranker damages an ordering that was already right.

**36% of the loss is out of reach of any reranker** — beyond depth 50, or never retrieved.
Reordering cannot touch it, and improving retrieval to reach it has been measured as
converting to nothing.

# 0.2 Recommendation

Stop. `0.9493` is the answer this corpus and these labels support. The residual is not
one problem with one fix; it is eighty-six separate hard questions with no shared
structure, twenty of which are never retrieved and one of which
(`9060`) has a gold document that appears nowhere in the 5,950 training questions.

---

# legalir4 — the residual is not a ranking problem

`PLAN-legalir3.md` argued that retrieval was the untouched half and would pay twice.
**That plan is falsified.** Experiment #9 improved candidate recall@5 by +0.0257 — three
times anything the reranker work produced — and the finished system moved by −0.0006
with 1,019 of 1,050 predictions identical. Retrieval and reranking are substitutes in
this pipeline; a better candidate order hands the reranker work it was already doing.

Do not train another embedder. One exists, it works, and it changes nothing downstream.

This document proposes the one direction the error inventory actually points at, and it
is cheap enough to test before committing to anything.

---

## 1. What the residual actually is

Ten interventions and two retrieval configurations land at 0.945 ± 0.006. The binding
constraint is the ~0.035 of gold that reaches the top-50 pool and cannot be placed in the
top five by any ordering of the same evidence.

The saved errors say why:

| query | what happened |
|---|---|
| 123452 | the amendment `226545` was candidate 5 and was dropped; the base decree `258342` it amends was kept |
| 132594 | gold `303485` was candidate 8; the output was five related tax-extension documents |
| 9060 | the sole gold was candidate 14, and it never appears as a positive anywhere in the 5,950 training questions |

The first two are the same failure. A decree and the decree amending it share most of
their text — that is what "amendment" means — so no cross-encoder separates them on text
alone, regardless of size, objective or training distribution. This is a
**distinguishability** failure, not a ranking one, and every intervention so far has been
a ranking one.

What *does* separate those documents is information the system currently discards:
document type, document number, issue date, and the relation between two candidates
sitting in the same shortlist.

---

## 2. Three diagnostics before any intervention

All three are CPU-only, run against files already in `_results/outputs` plus the corpus,
and take minutes. **Run these first.** If they come back weak, the intervention below is
not worth building, and that will have cost an afternoon rather than a rental.

### D1 — do questions cite documents explicitly?

Vietnamese legal questions frequently name a document: *"theo Nghị định 15/2020/NĐ-CP"*,
*"Điều 5 Luật Xây dựng 2014"*. `store.py` already extracts document numbers into chunk
labels, so an exact match is a lookup, not an inference.

Measure, over the 1,050 validation questions:

- what fraction contain a document-number or year pattern
- among those, how often the gold document's number appears in the question text
- **precision**: when a candidate's number appears in the question, how often is it gold

A feature is worth having if that last number is high and it fires on more than ~5% of
questions. High precision on a small slice is exactly what a 12-weight linear stacker can
exploit and what a cross-encoder cannot learn from 5,950 examples.

### D2 — are the wrong answers siblings of the right one?

For each of the 86 missing gold documents, check whether the five predicted documents are
structurally related to it: same document-number stem, same issuing body, one amending
the other, adjacent issue dates.

If the failures are dominated by siblings, the problem is tie-breaking among
near-duplicates and the features in §3 address it directly. If the failures are
unrelated documents, they are ordinary retrieval misses and this whole direction is
wrong.

### D3 — are multi-gold answers related to each other?

Multi-gold questions score 0.7814 against 0.9613 for single-gold. Check how often the
gold documents of a multi-gold question are structurally related — a law and its
amendment, a decree and its implementing circular.

If they usually are, then relatedness is *evidence about relevance*, and a candidate
becomes more likely to be gold when a related candidate already scores well.

---

## 3. The intervention: structural features for the stacker

Not a new model. Five to eight features added to `stack.py`, computed from the store and
the candidate list alone.

```
cite_exact      the candidate's document number appears in the question
cite_year       the candidate's issue year appears in the question
type_match      question says "nghị định"/"thông tư"/"luật" and the candidate is one
sibling_best    best reranker score among candidates structurally related to this one
is_amendment    another candidate in this shortlist is the document this one amends
is_amended      the reverse relation
recency_rank    rank by issue date among this shortlist's related group
```

`legalir2/structure.py` already recovers the Chương / Mục / Điều hierarchy and the store
carries document numbers in chunk labels, so the parsing exists. What is new is exposing
it to the ranker.

**Why this is not the MMR idea that was already rejected.** `ANALYSIS-v7` showed
correctly that Recall is additive over the selected set, so greedy selection by expected
contribution is already optimal and a diversity *bonus* is unjustified. These features do
something different: they improve the probability *estimate* for each document. That "B
amends A, and A scores highly" is evidence about whether B is relevant is a fact about
the world, not a preference for variety. The selection rule stays greedy.

---

## 4. Why this can be tested for free

The frozen score cache stores the reranker's raw per-passage scores, and every feature
above depends only on document ids and the store — not on the reranker. So the entire
experiment runs offline:

1. extend `build_features` with the structural features
2. refit the stacker on the cache, **out of fold**, with folds keyed on normalised
   question text (`cv --fit-stacker --questions ... --stacker-folds 5`)
3. `cache.evaluate()` replays the ranking and `compare.py` gives the paired interval

No GPU, no training, minutes per iteration. If no cache exists on the box, building one
is a single scoring pass; after that, every variant of the feature set is arithmetic.

That is the cheapest experimental loop this project has ever had, and it is the right
place to spend the next effort precisely because the expensive loops have been exhausted.

---

## 5. What success looks like, honestly

Recoverable-by-reordering mass is 0.0257 in single-gold questions and 0.0094 in
multi-gold. If structural features fix a third of the sibling confusions, that is roughly
**+0.008** — right at the detection floor of 0.0094 on 1,050 questions.

So: judge this on the **7,000-question cross-fitted cache**, where the floor is 0.0036,
not on the 1,050 holdout. That is what the cache was built for and it has never been used
for something with a mechanism behind it.

One reason to expect better than the average: `cite_exact` should be nearly deterministic
when it fires. A feature with high precision on a small slice moves more than its
frequency suggests, and D1 measures exactly that before anything is built.

And one reason to expect nothing: query `9060`'s gold never appears in the training data
at all. Some share of the residual is questions whose answer is simply not learnable from
these labels, and no feature reaches those.

---

## 6. Order of work

1. **D1, D2, D3** — an afternoon, no GPU, no rental. Decide on the evidence.
2. If D2 shows sibling-dominated failures: implement §3, refit on the cache, measure.
3. If D1 shows high-precision citation matches: `cite_exact` alone is worth shipping even
   if the rest measures null.
4. If all three come back weak: the residual is unlearnable from these labels, 0.9493 is
   the answer, and that conclusion is itself worth writing down.

Nothing here needs a GPU until step 2, and step 2 needs one scoring pass rather than a
training run. That is deliberate: every expensive hypothesis in this project has been
tested, and the ones that remain are cheap.
