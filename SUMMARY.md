# DSC-2026 Task 1 (LegalIR) — state of the system

*Written 16 September 2026. Covers everything from the first baseline to the current
0.9508 submission and the one experiment still on the table.*

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

Public tracked validation at roughly **+0.004** on v4 and v9. v10 is the first submission
whose public score moved against the intervention's premise; see §8.

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
| Out-of-fold stacker weights (`v10`) | **−0.0044 public.** Confounded by reranker strength, but it rules out trusting the fusion prior more. §8. |

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

## 9. Outstanding security item

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

## 10. Repository map

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
| `PROCESS-v6.md` | architecture and the substitution finding |
| `PLAN-legalir4.md` | the three structural diagnostics and their negative verdict |
| `RUNBOOK-v6.md` | operating instructions for a fresh rental |

**Two interpreters.** `.venv` — transformers 4.51.3, the shipped pipeline.
`.venv5` — transformers 5.17.0 + flash-linear-attention, Qwen3.5 only. Do not upgrade
`.venv` in place; the shipped pipeline is pinned for a reason.

---

## 11. Honest bottom line

0.9508 is a good score, obtained by finding and fixing a bug that was discarding a second
reranker, not by a new idea. The remaining 0.049 is eighty-six separate hard questions
with no shared structure — twenty never retrieved at all, and at least one
(`9060`) whose gold document appears nowhere in the 5,950 training questions.

Twelve interventions now sit inside **0.945 ± 0.006** on validation. Every mechanism with
a story behind it has been tested: better retrieval (substitutes, §4), a second reranker
(the one win, §3), structural metadata (§5), and re-weighting the ranking stage (§8).
The last of those was the final addressable bucket, and it did not move.

**Stop here.** 0.9508 is what this corpus and these labels support. The gap to the top of
the board is 0.0083 — smaller than the noise on a single validation comparison — and
nothing left on the list has honest positive expected value against another rental.
