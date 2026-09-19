# DSC-2026 Task 1 (LegalIR) — state of the system

> **Active submission target, 19 September 2026: private test only.** The user supplied
> `private-official.json`: 2,080 questions with hidden answers and no public-ID overlap.
> Use separate private predictions/caches. Public scores and submissions below are
> historical; they are not private results. See [PRIVATE-TEST.md](PRIVATE-TEST.md).
> The completed 49-miss evidence audit is at
> [miss-audit/REPORT.md](../../analysis/v11/investigation/miss-audit/REPORT.md).
> Separate frozen V9 and duo private inference is ready via `scripts/gpu/gpu-private-h100.sh`;
> see [RUNBOOK-private.md](../runbooks/RUNBOOK-private.md). The user has now supplied both private
> prediction JSONs and reported **duo 0.9469 vs V9 0.9457** (+0.0012).
> Duo is the current best reported private baseline. Both files cover all 2,080 IDs;
> comparison details are in `analysis/private/leaderboard-comparison.json`.

*Written 16 September 2026, updated 17 September. Covers everything from the first
baseline to the 0.9508 submission that remains the best, and the fourteen interventions
that did not beat it.*

---

## 1. Where things stand

| | |
|---|---|
| **Best public score** | **0.9508** recall, 0.2054 precision |
| **Leaderboard position** | 9th |
| **Top of board** | 0.9591 |
| **8th / 7th / 6th** | 0.9517 / 0.9535 / ~0.9548 |
| **Deployed configuration** | `v9` — two rerankers under a 20-feature stacker |
| **Parameter budget** | 1.73 B of the 4 B limit |
| **Submission artefact** | `outputs/submission-v9.zip` |

### Score progression

| version | what changed | validation | public |
|---|---|---|---|
| v0 | lexical-only baseline | 0.8483 | — |
| v1 | dense + lexical fusion, tuned | — | 0.9223 |
| v2 | cross-encoder reranker | — | 0.9433 |
| v4 | 12-feature learned stacker | 0.9454 | 0.9493 |
| **v9** | **two rerankers, 20-feature stacker** | **0.9464** | **0.9508** |
| v10 | out-of-fold stacker weights | — | 0.9464 (−0.0044) |
| v11 | bundled 2,048-token evidence as a stacker feature | 0.9463 (±0.0000) | 0.9498 (−0.0010) |
| v12 | the 1.88 B reranker in place of g32 | 0.9444 (−0.0016) | 0.9449 (−0.0059) |

Public tracked validation at roughly **+0.004** on v4 and v9. Everything after v9 is a
failed attempt to beat it: v10 moved against its own premise (§8), v11 did not move at all
(§9), and v12 spent 3.05 B of the 4 B budget to lose 0.0059 (§10). **v9 remains the
submission.**

---

## 2. Architecture

```
question
  │
  ├── dense retrieval      models2/embedding (0.57 B, fine-tuned)
  └── lexical retrieval    BM25F + bigram, tuned weights
        │
        └── reciprocal-rank fusion  →  top-50 candidates
              │
              ├── reranker 1   models2/reranker-g32   (0.57 B, fine-tuned, group-size 32)
              └── reranker 2   partner_qwen           (0.60 B, off the shelf)
                    │
                    └── stacker: 20 features, listwise softmax, linear
                          │
                          └── top 5 documents
```

**Stacker features.** Twelve shared features (fusion rank, dense rank, lexical rank, and
per-reranker max / second / mean / spread / near-top), plus a five-feature block for the
second reranker, plus three agreement features (`rr_agree_rank`, `rr_gap`, `rr_min_norm`).
All min–max normalised **per question** before the fit.

**Constraints honoured throughout.** Total deployed parameters 1.73 B against a 4 B
ceiling. No models outside the approved sheet. No external API calls, commercial or
otherwise.

---

## 3. The result that mattered, and why it was nearly missed

Ensembling `partner_qwen` with `reranker-g32` measured **+0.0086 [+0.0011, +0.0163],
p = 0.035** — the only paired interval in ten interventions that excluded zero. But when
a stacker was switched on, the gain collapsed to +0.0003 with 1,030 of 1,050 predictions
identical.

That was not redundancy. `build_features` read `passage_scores[0]` and nothing else, so
the moment a stacker loaded, reranker two was scored, paid for, and thrown away. The
ranking it produced was the single-model ranking — which is exactly what 1,030 identical
predictions means.

`stack.py` now emits a feature block per reranker, and `Stacker.order` raises on a weight
vector whose width does not match the pipeline rather than silently ignoring a model.
Re-run properly, the two-reranker stacker gave **+0.0010** on validation and **+0.0015**
on public. Small, but real, and the mechanism is understood.

---

## 4. The ceiling, and the mechanism behind it

Eleven interventions and two retrieval configurations all land at **0.945 ± 0.006** on
validation. The reason is not that the interventions were bad.

### Retrieval and reranking are substitutes

Experiment #9 produced a tuned embedder worth **+0.0257 candidate recall@5** — three
times anything the reranker work ever produced. The finished system then moved
**−0.0006**, with 1,019 of 1,050 predictions identical.

Better candidates hand the reranker work it was already doing. Conversion from candidate
recall to final recall runs at about 58 %, and it falls as candidates improve. **Do not
train another embedder.** One exists, it works, and it changes nothing downstream.

### The residual is diffuse

Of the 86 gold documents missing from the final five:

| where the gold document was | count | recall lost | share |
|---|---|---|---|
| already in candidates 1–5 | 14 | 0.0089 | 16 % |
| 6–10 | 10 | 0.0071 | 13 % |
| 11–20 | 15 | 0.0113 | 21 % |
| 21–50 | 11 | 0.0078 | 14 % |
| 51+ (beyond rerank depth) | 16 | 0.0076 | 14 % |
| never retrieved at all | 20 | 0.0119 | 22 % |

Three conclusions:

- **No bucket holds more than 22 %,** and no structural pattern explains any bucket.
  There is no single lever, which is why every intervention moved about the same 0.008.
- **0.0089 is actively destroyed** — fourteen gold documents were inside the *fusion's*
  top five and the ranking stage pushed them out.
- **36 % of the loss is out of reach of any reranker** (beyond depth 50, or never
  retrieved), and improving retrieval to reach it has been measured as converting to
  nothing.

---

## 5. What was tested and did not work

| hypothesis | verdict |
|---|---|
| Better retrieval → better final score | **Falsified.** +0.0257 candidate recall → −0.0006 final. |
| Questions cite the statute they want | **Dead.** 5 of 1,050 questions (0.5 %) contain a document number. |
| Failures are sibling/amendment confusion | **Real but small.** 47.4 % of failures have a related document in the top five — against a 42.1 % control. |
| Multi-gold answers are structurally related | **2.5× lift on 19 pairs.** Far below the detection floor. |
| Public test is multi-gold heavy | **Wrong.** `5×precision − recall` is 0.0766 on validation and 0.0767 on public — identical structure. |
| More data, base init, deeper negatives (`submission-final`) | **−0.0162.** Four things changed at once; described as "same recipe, more data", and it wasn't. |
| MMR / diversity bonus | **Unjustified.** Recall is additive over the selected set, so greedy selection is already optimal. |
| Bundled joint evidence for the partner (`v11`) | **Better feature, same score.** Weight 0.242, 55 changed top-five sets, 0 changed recall. §9. |
| Out-of-fold stacker weights (`v10`) | **−0.0044 public.** Confounded by reranker strength, but it rules out trusting the fusion prior more. §8. |
| A 3.3× larger reranker (`v12`) | **−0.0016 validation, −0.0059 public.** Capacity is not the binding constraint. §10. |

---

## 6. Measurement bugs found and fixed

Two of these were silently inflating or flattening results for most of the project.

1. **In-sample stacker fitting.** `cv --fit-stacker` fitted and evaluated on the same
   cached questions. It now fits *k* fold stackers, reports out-of-fold, and writes
   all-data weights with an `out_of_fold` record attached.
2. **Folds keyed on question id.** The labelled set contains **16 duplicate question
   groups**; keying on id put copies of the same question on both sides of the split —
   precisely the leak cross-fitting exists to prevent. Folds are now keyed on normalised
   question *text*.
3. **`build_features` reading only `passage_scores[0]`** — §3 above.
4. **`GROUPS` as a bash variable.** Bash owns that name as a special array of the current
   user's group ids, so the assignment lands in element 0 and `$GROUPS` expands to a gid.
   It silently skipped a whole stage in `gpu-phase8.sh`. Renamed `DOC_GROUPS` everywhere.

---

## 7. Statistical ground rules established

- **Detection floor ≈ 0.0094** at n = 1,050 with a paired bootstrap plus exact sign test.
  This is not a universal constant — it depends on the distribution of per-question
  differences, and is ~0.0036 on the 7,000-question cross-fitted cache.
- **"Identical predictions" counts equal recall, not equal output.** 1,045 / 1,050
  identical in the ensemble run meant equal *scores*, not the same five documents.
- **Winner's curse applies twice** — once inside a sweep, once in picking the best
  leaderboard submission. Do not read a leaderboard gap under 0.015 as evidence.
- **Scorer semantics**: `precision = |A∩G| / len(submitted)`, 1–5 answers are legal, and
  **the ranking is on recall alone**. Submitting five is always weakly optimal for rank.

---

## 8. The out-of-fold stacker — run, and it failed

> **RESULT: 0.9464 public, −0.0044 against v9.** Reverted. v9 remains the submission.
> The experiment did not cleanly test its hypothesis — see "the confound" below — but it
> killed a larger idea than the one it was aimed at.

**`gpu-calibrate.sh`** — ~3.8 h, ~$1.40, called at **odds roughly one in three**.

`v9-stacker-ens.json` was fitted on 2,000 training questions — the same questions
`reranker-g32` was fine-tuned on. On its own training data a reranker separates gold from
negatives far more cleanly than on anything unseen, so the fit over-trusts it:

```
rr_max      +1.5556      the reranker
fused_rr    +0.2874      the fusion's own ranking
```

At inference those weights are wrong in a specific direction, and the loss map names the
consequence: **the 0.0089 of recall where gold was already in the fusion's top five and
the ranking stage pushed it out.**

The fix is out-of-fold fitting: train a reranker on fold A, score fold B with it, fit the
stacker on those features. The weights then describe how a reranker behaves on questions
it has never seen — the only situation that ever occurs at inference. The **shipped**
rerankers are still what gets deployed; only the weights come from out-of-fold data.

| validation gain | public ≈ | place |
|---|---|---|
| +0.001 | 0.9518 | 8th |
| +0.003 | 0.9538 | 7th |
| +0.004 | 0.9548 | 6th |

It returned **−0.0044**. Phase five's earlier cross-fitted measurement of −0.0014 — which
was dismissed because both bugs in §6.1 and §6.2 were live at the time — turns out to
have had the sign right.

### The confound

`reranker-foldA` differs from `reranker-g32` in **two** ways, not one. It never saw fold
B's questions, and it was trained on **half the data**. So it is not merely an honest
reranker — it is a *weaker* one.

The stacker did what it should: it learned to trust a weaker model less. Those weights
were then deployed on the full-strength model, which systematically under-weights
`rr_max` for a reason that has nothing to do with memorisation. That alone predicts the
drawdown, so the memorisation hypothesis was never isolated.

A clean version would hold reranker strength fixed and vary only what the *stacker* saw —
k-fold rerankers all trained on the same number of groups, or cross-fitting the stacker
alone over cached scores from the shipped model. Neither was run, and neither is worth a
rental now, because of what follows.

### What it killed anyway

The premise behind the 0.0089 "actively destroyed" bucket (§4) was that on those 14
questions the *fusion* was right and the reranker overrode it — so shifting weight from
`rr_max` toward `fused_rr` should recover them.

Weight moved that direction and the score fell. Whatever is recovered on those 14
questions costs more than it gains everywhere else. **That bucket is not reachable by
re-weighting, and the whole family of shrink-toward-the-fusion-prior ideas goes with it.**

That is a bigger result than the one the experiment was aimed at, and it is the reason to
stop rather than to iterate.

---

## 9. v11 — the ranking stage moves documents that are never gold

> **RESULT: 0.9498 public (−0.0010); validation recall unchanged on every one of 1,050
> questions.** The clearest measurement in the project, and the one that finally explains
> the shape of the ceiling rather than just reporting it.

The hypothesis: pack up to six retrieved passages per candidate into one 2,048-token
prompt so the Qwen partner attends *across* them, instead of scoring three passages
independently and max-pooling. Added as one extra stacker feature beside the existing
twenty, with a matched control — the same twenty features refitted on the same cache —
so that refitting could be separated from new evidence. v10 had no such control.

### The feature works

| | control | treatment | shift |
|---|---|---|---|
| `rr_max_2` | 0.1113 | 0.0663 | −0.045 |
| `rr_second_2` | 0.1018 | 0.0623 | −0.040 |
| `rr_mean_2` | 0.1142 | 0.0748 | −0.039 |
| **`qwen_bundle_score_v1`** | — | **0.2422** | — |

The stacker pulled ~0.12 of weight out of Qwen's three passage-level features and put
0.242 on the single bundled score — **more than all three of them combined**. Reading one
joint 2,048-token view of a document is a better use of that model than max-pooling three
768-token passages. That half of the hypothesis is confirmed.

### And it changes nothing

| comparison | orderings changed | top-five sets changed | **recall changed** |
|---|---|---|---|
| frozen → control | 12 | 3 | **0** |
| frozen → treatment | 115 | 55 | **0** |
| control → treatment | 115 | 56 | **0** |

Fifty-five questions came back with a *different set of five documents* and every one of
them scored identically. The reshuffling is real, substantial, and happens entirely among
documents that are not gold.

Public agrees. 0.9508 → 0.9498 across ~960 questions is **−0.00104**, or exactly one
single-gold question. So the treatment alters roughly one prediction in a thousand: none
of 1,050 on validation, one of ~960 on public, and that one got worse.

### Why this is the real ceiling statement

Everything before said *the residual is diffuse* (§4). This says something stricter: the
part of the ranking a better feature can still move is **~98–99% non-gold**. Zero gold
touched in 55 set changes puts the gold-touch rate near 1–2%.

That follows from the residual map. Only 14 of 86 missed gold documents sit in candidates
1–5 and 10 more at 6–10; the position-5/6 boundary — the only place reordering has any
leverage — is populated by sibling documents, which `PLAN-legalir4` D2 measured crowding
the top five of 42% of questions the system already gets *right*. A better ranker
rearranges those siblings. There is no gold underneath them to find.

**A good ranking feature is not the same as a better score, and this is why.**

### One thing worth keeping

`control_vs_frozen` was 0.0, with `rr_max` refitting to 1.5552 against v9's 1.5556. The
pipeline reproduced the shipped stacker almost exactly from a freshly collected cache.
After v10, that reproducibility is worth having on the record.

---

## 10. v12 — capacity is not the binding constraint

> **RESULT: −0.0016 validation (CI [−0.0105, +0.0071], p = 0.864); 0.9449 public.**
> Public landed inside the predicted interval. The measurement was right and the bet lost.

`reranker-2b-tuned` (1.88 B) replaced `reranker-g32` (0.57 B) beside `partner_qwen`, under
the same 20-feature stacker. Deployed budget 3.05 B against v9's 1.73 B — the first
configuration to use most of the 4 B allowance.

| | validation |
|---|---|
| control: g32 + qwen, same interpreter | 0.9460 |
| treatment: 2b + qwen | 0.9444 |
| paired difference | **−0.0016**, 16 better / 18 worse / **1,016 identical** |

### The weights predicted it three hours early

Four disjoint 500-question chunks, fitted independently:

| | chunk 0 | 1 | 2 | 3 | mean | v9 (g32) |
|---|---|---|---|---|---|---|
| `rr_max` (model 1) | 1.008 | 0.959 | 1.011 | 0.963 | **0.985** | **1.555** |
| `den_rr` | 1.159 | 1.060 | 1.161 | 1.177 | **1.139** | 0.616 |
| training loss | 0.788 | 0.748 | 0.787 | 0.762 | 0.771 | 0.545 |

`rr_max` within ±0.03 across four independent samples: the 2B earns **63% of the weight
g32 earned**, and dense retrieval nearly doubles to compensate. The loss says the same —
fitting on ~490 questions should be *easier* than on ~2,000, yet the loss is 40% higher.

**The chunk weight table is a genuine early readout.** It cost nothing, arrived before the
measurement, and called the result correctly. Worth doing again.

### What went wrong on the way, and what it taught

**The dense index was missing and nothing said so.** `predict` and `stack` do
`dense_directory=args.dense if (Path(args.dense)/"manifest.json").is_file() else None` —
silent fallback to lexical-only. The control came back at 0.9219 against v9's 0.9463 and
read as a transformers-5 regression; it was the index. The giveaway was candidate recall
matching `v2-lex-tuned` to four decimals at all six depths. `gpu-arm-a.sh` now hard-fails
preflight on a missing or store-mismatched index, and `control` refuses to report if
candidate recall @50 comes back at the lexical 0.9608 instead of the fused 0.9805.

**transformers 5 was innocent.** With the index restored the control reproduced v9 —
0.9460 against 0.9463, tokenizer ids identical under 4.51.3 and 5.17.0. The control stage
earned its 30 minutes twice over: once by catching a broken environment, once by clearing
one.

**flash-linear-attention is worth 3.2×, not the 1.35× predicted.** Measured on the same
box, same bench: 67.9 → 164.8 p/s at batch 8, 55.6 → 216.5 at batch 64. Without it the
reference `chunk_gated_delta_rule` is a Python loop whose throughput *falls* as batch
grows; with it, throughput rises. Advising against installing it cost most of a day.

---

## 11. Outstanding security item

**`aic-readwrite.conf` contains a Google OAuth refresh token with `scope = drive`** — full
read/write access to the entire Drive account. It was committed to the public repository
`github.com/meocatmeo/DSC-2026`. The history was purged and force-pushed, but **the
credential itself has never been rotated**, and anyone who cloned or cached the repo
before the purge still holds a working token.

Fix, in order:

```
rclone config reconnect gdrive:        # re-authorise with scope = drive.file
```

Then revoke the old grant at <https://myaccount.google.com/permissions>.

`.gitignore` now covers `*.conf`, `rclone.conf`, and `rclone.exe`, and `_upload/` excludes
them — but that prevents the *next* leak, not this one.

---

## 12. Repository map

| path | what it is |
|---|---|
| `legalir2/stack.py` | feature construction, `Stacker`, multi-reranker support |
| `legalir2/cache.py` | frozen score cache, `folds()` keyed on normalised text |
| `legalir2/retrieval.py` | candidate recall per channel at depth, paired bootstrap |
| `legalir2/mine.py` | hard-negative mining, `--channel {fused,dense,lexical}` |
| `legalir2/train_ce.py` | cross-encoder training, per-question score standardisation |
| `gpu-ensemble.sh` | **produced the current 0.9508 submission** |
| `gpu-calibrate.sh` | the out-of-fold stacker experiment (§8) |
| `gpu-legalir3.sh` | staged retrieval work — falsified, kept for the diagnostics |
| `docs/history/PROCESS-v6.md` | architecture and the substitution finding |
| `docs/history/PLAN-legalir4.md` | the three structural diagnostics and their negative verdict |
| `docs/history/RUNBOOK-v6.md` | historical operating instructions; current private runs use `RUNBOOK-private.md` |

**Two interpreters.** `.venv` — transformers 4.51.3, the shipped pipeline.
`.venv5` — transformers 5.17.0 + flash-linear-attention, Qwen3.5 only. Do not upgrade
`.venv` in place; the shipped pipeline is pinned for a reason.

---

## 13. Honest bottom line

0.9508 is a good score, obtained by finding and fixing a bug that was discarding a second
reranker, not by a new idea. The remaining 0.049 is eighty-six separate hard questions
with no shared structure — twenty never retrieved at all, and at least one
(`9060`) whose gold document appears nowhere in the 5,950 training questions.

Fourteen interventions now sit inside **0.945 ± 0.006** on validation. Every mechanism
with a story behind it has been tested: better retrieval (substitutes, §4), a second
reranker (the one win, §3), structural metadata (§5), re-weighting the ranking stage (§8),
richer joint evidence (§9), and more model capacity (§10).

§9 and §10 together close it. A feature can be genuinely good — the stacker gave the
bundled score more weight than all three of the partner's passage features combined — and
still not move the score, because **the region of the ranking it can affect contains
almost no gold documents**: 55 changed top-five sets, zero changed recall. And a model can
be 3.3× larger and do no better, because the constraint was never capacity. Every
remaining idea in this project is a ranking idea, and that is what a ranking idea buys.

One family is untested: adding a *third* independent reranker (`partner_viranker`
measured +0.0031 ensembled with g32) rather than replacing one. It is the same shape as
the only intervention that ever worked. After v12 I would not fund it.

**Stop here.** 0.9508 is what this corpus and these labels support. The gap to the top of
the board is 0.0083 — smaller than the noise on a single validation comparison — and
nothing left on the list has honest positive expected value against another rental.
