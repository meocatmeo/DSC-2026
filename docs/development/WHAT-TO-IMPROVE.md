# What to improve — measurement limits and the one remaining mechanism

*21 September 2026. Read after [development/README.md](README.md). Nothing here is a
measured improvement; the two numbered findings are arithmetic on existing artifacts.*

---

## 0. The headline

Your instinct that development stopped predicting private is right, but the diagnosis
is sharper than "not linear":

> **Development at n = 429 resolves ±0.0086. The gains you promoted on were +0.0012 and
> +0.0018 — 0.3σ and 0.4σ. There was no signal to be non-linear.**

The deep-fusion promotions were selection on noise, and selection on noise is
systematically harmful, not neutral. That alone explains the regressions without
needing distribution shift, calibration drift, or overfitting of the 23 coefficients.

---

## 1. Resolving power of every set you have

`φ` = P(recall changes | answer set changes) = **6/259 = 2.3%**, measured on development
from the 256-vs-baseline comparison. Paired SE = √(changed × φ × E[d²] / n²).

| comparison | n | changed sets | Δ | SE | σ |
|---|---:|---:|---:|---:|---:|
| development: 256 vs full-text | 429 | 259 | +0.0012 | 0.0044 | **0.3** |
| development: 512 vs full-text | 429 | 254 | +0.0018 | 0.0044 | **0.4** |
| private: full-text vs duo | 2,080 | 911 | +0.0041 | 0.0017 | 2.4 |
| private: 256 vs full-text | 2,080 | 1,223 | −0.0046 | 0.0020 | 2.3 |
| private: 512 vs full-text | 2,080 | 1,219 | −0.0051 | 0.0020 | 2.6 |

95% resolution for a change touching ~60% of answers:

| set | n | ± |
|---|---:|---:|
| development | 429 | **0.0086** |
| validation | 1,050 | 0.0055 |
| **both pooled** | **1,479** | **0.0047** |
| private | 2,080 | 0.0039 |

Two consequences.

**Development could never have seen these effects.** It is 7× too coarse. The 512 run's
+0.0006 over 256 is 0.1σ. Treating that as evidence was not optimism, it was reading a
number that does not exist.

**Your private results are real but marginal.** 2.3–2.6σ, and σ rests on a six-event
estimate of φ. If φ is three times larger, everything collapses to ~1.4σ. So:
0.9510 is probably the best of the five, and the deep-fusion runs probably did regress —
neither is established. Do not spend further runs separating 0.9457 from 0.9510.

---

## 2. The system has converged

Comparing the three distinct private submissions (manual-01 is a duplicate of duo —
99.9% identical sets, Jaccard 1.000):

| documents unanimous across v9, duo, full-text | questions | |
|---|---:|---:|
| 5 of 5 | 792 | 38.1% |
| 4 of 5 | 1,020 | 49.0% |
| 3 of 5 | 245 | 11.8% |
| ≤2 of 5 | 23 | 1.1% |

**Mean unanimous core: 4.24 of 5 documents.** Mean distinct documents proposed across all
three: 5.83. Every method you have is arguing over **0.76 slots per question**.

And those slots are mostly not gold. Full-text duo changed **911 answer sets** against duo
to produce a net **+8.5 questions** of recall — about a 1% conversion. Deep fusion changed
**1,223 sets** for **−9.6 questions**. This is the v11 finding at private scale: the
region the ranking stage can still move contains almost no gold documents.

---

## 3. Why the deep-pool experiment regressed, mechanically

From `PIPELINE.md`:

> *Preserving the original candidate pool does not preserve its top-five answers. The
> learned prior reranks the whole pool.*

That is the bug in the experiment design, not in the code. It bundled two changes:

1. **Add up to four candidates** — coverage 0.9666@50 → 0.9864@100 → 0.9887 full pool.
2. **Refit the prior over the entire enlarged pool** — which re-ranked everything.

Change 1 can only help: a candidate that was never there cannot lose you a gold document.
Change 2 is a coin flip on 59% of your answers, and §2 says those flips land on non-gold
documents. You paid for the downside of (2) to get the upside of (1), and the bottleneck
report confirms the upside never arrived — *"none reached the submitted top five"*.

---

## 4. The one remaining mechanism with >0.01 of headroom

| stage | coverage |
|---|---:|
| full scored pool | 0.9887 |
| final five | 0.9520 |
| **conversion** | **96.3%** |

Ranking converts the pool at 96.3%. Eleven prior interventions say that number is at its
ceiling. So the only lever left is the pool itself: at an unchanged 96.3% conversion,
taking coverage from 0.9887 toward 1.000 is worth about **+0.011** — the first candidate
effect in this project that would clear its own measurement floor.

It is also the hardest. `RETRIEVAL-EXPANSION.md` measured citation expansion at +0.0040
candidate recall, below floor, and found 31 of 34 unreachable gold documents already share
most of their question's vocabulary. The remaining 0.0113 is not a vocabulary problem.

**So the honest position is: no cheap change clears the bar.**

---

## 5. Recommendation

**Keep the 0.9510 full-text duo. Do not submit another variant of the current method.**
Five private submissions span 0.0053, about 2σ. More draws from that distribution do not
improve your standing; if the final ranking uses a separately held split, picking the
maximum of five noisy draws regresses by roughly 0.003.

**Before any further experiment, register a minimum detectable effect.** Pool development
and validation (1,479 questions, ±0.0047) and require the paired interval to exclude zero.
No change tested since v9 would have passed that gate. That single rule would have
prevented both deep-fusion runs.

**If you run one more thing, make it append-only.** Take the deep pool's coverage gain
without its ranking churn: a new candidate may enter the top five only when it beats the
member it displaces by a calibrated margin, with the baseline ranking otherwise frozen.
Tune the margin on the pooled labeled set so that fewer than ~100 of 2,080 answer sets
change. That bounds the downside to roughly 100 × 2.3% × 1 ≈ 2 questions while preserving
whatever the added candidates are worth. It reuses cached scores, needs no GPU training,
and — unlike everything since v9 — has a failure mode you can size in advance.

---

## 6. What I could not check

- Private gold labels do not exist locally; every private number here is user-reported.
- φ = 2.3% rests on six events. Recomputing it on the 1,050-question validation set would
  tighten every σ in §1 and costs nothing.
- The 256/512 private prediction JSONs were not in `submissions/`, so §2's consensus
  analysis covers three configurations, not five. Adding them would sharpen the core count.
