# Fixed-probe check, current-component fusion and a fourth voter

21 September 2026. **No submission is promoted.** The exact old probe fails
to replicate positively on development. A separate current-component fusion
gains one development question, below the required pooled score gate.

## 1. Exact policy supplied by the user

Source: `scripts/analysis/rank_fusion_probe.py`. This file was left unchanged.
The voters are historical V9, g32-only, Qwen-only and bundle-only orderings.
The formula is reciprocal-rank fusion with K=60, V9 weight W, and weight 1
for each other voter; ties prefer the V9 ordering. W is selected from
`1,2,3,4,5,6,8,10,14,20` using validation only, then frozen before development
labels or orderings are loaded by the new evaluation wrapper.

Reproduced validation cross-fit: **+0.00284127**, across the exact 20 repeated
two-way half splits used by the supplied script. This is below +0.0052.
The approximately +0.0051 in-sample gain remains a biased selection result.
Repeated splits are not independent replications or a confidence interval.

Frozen validation winner: **W=4**. All 429 development questions are present
in the historical training cache and were matched by ID, text and labels.
The wrapper checks for development/validation ID and normalized-text overlap.

| Fixed old probe on development | Result |
| --- | ---: |
| Historical V9 baseline recall | 0.948523699 |
| W=4 fusion recall | 0.944444444 |
| Paired difference | **-0.004079254** |
| Better / worse questions | 1 / 2 |
| Changed top-five sets | 237 / 429 |
| Paired bootstrap 95% interval | [-0.011655012, +0.001165501] |

Thus there are **not two independent positives**. Development is disjoint from
this fusion-weight selection, but both sets have extensive historical reuse,
and its underlying component scores come from the old training cache. It is
not an untouched end-to-end holdout. Do not tune W on these development results
and then call the resulting gain a confirmation.

Report: `reports/current-fusion/20260921-fixed-probe-v2/report.json`.
Registration and `frozen-policy.json` are alongside it.

## 2. Fusion using the current components

`legalir2/current_fusion.py` reconstructs complete orderings using the actual
cached scores, not the input candidate order or submitted top-five lists:

- V9: frozen `features1 @ weights`, with the original document-ID tie rule;
  the top five must replay the short-duo cache's V9 baseline.
- Short duo: `prior + 0.25 * (ability - prior)` over its entire scored pool.
- Full-text duo: the same frozen blend over its own full-text scored pool;
  top five must replay `fulltext_prior.predict(..., BASELINE)`.

Question IDs, text and gold are aligned across voters; saved cache identities,
dataset hashes and full-text calibration score hashes are verified. Missing
documents contribute **zero** to RRF; they are not assigned an invented tail rank.

The separately registered first policy is K=60, weights 1/1/1, ties prefer
the full-text ordering then document ID. This is a new design choice, not the
historical probe's W=4 policy and not three independent neural models: V9 is
already part of both duo priors, and short/full-text duo share model weights.
No search over current-component weights was performed.

| Current component / fusion on 429 development questions | Recall |
| --- | ---: |
| V9 | 0.949689200 |
| Short duo | 0.951437451 |
| Full-text duo reference | 0.950271950 |
| Full-order three-voter fusion | **0.952602953** |
| Top-five-only fusion, diagnostic | 0.952602953 |

Full-order fusion gains **+0.002331002**, one better question and zero worse,
while changing 113 answer sets. The paired bootstrap interval is
`[0.0, 0.006993007]`. The top-five-only diagnostic gains the same question but
changes 78 sets. Deeper ranks add changes here, **not additional measured recall**.
Mean available depths are 50.00 / 16.01 / 17.40; mean unanimous top-five core
on development is 4.256 documents. These are development numbers, not the
user's private overlap measurement.

Historical V9 and current V9 differ slightly because these are different
frozen scoring caches. Report each fusion against its own replayed baseline.
Do not silently substitute the historical bundle-only voter for full-text duo.

Report and full ordering export:
`reports/current-fusion/20260921-current-v2/{report,development-rankings}.json`.
The report also contains per-component paired comparisons and depth statistics.
No actual full-text validation score cache is available locally. Therefore the
current three-voter fusion has **not passed the pooled 1,479-question gate**.
Reconstructed validation candidate coverage cannot replace those predictions.

## 3. Optional fourth voter: ViRanker

**Recommendation after CPU sensitivity checks: keep the fourth voter out of
production.** Unused parameter budget is capacity to experiment, not evidence
that another vote improves predictions. The scorer below is an optional labeled
experiment only; it has not changed the baseline or created a submission.

Added `legalir2/fourth_voter.py` for a labeled GPU experiment. It retains V9,
short duo and full-text duo and scores the union of their full candidate pools
with the off-the-shelf `namdp-ptit/ViRanker` checkpoint. That repository is
listed as `partner_viranker` in the local `APPROVED` mapping. Other bge/jina
checkpoints are not enabled without checking their exact approved identities.

ViRanker's [published configuration](https://huggingface.co/namdp-ptit/ViRanker/blob/main/config.json)
uses XLM-R. The old registry comment calling it PhoBERT lineage is inaccurate.
Its [model card](https://huggingface.co/namdp-ptit/ViRanker) identifies bge-m3
ancestry and single-logit pair scoring. Different training may provide useful
error diversity, but architecture alone does not establish that.

Historical validation files give g32 solo 0.937142857, g32+ViRanker 0.940158730
(+0.003015873), and g32+Qwen 0.945714286. The ViRanker result supports testing
complementarity against that old weaker baseline; it does not establish a
marginal gain on today's V9/Qwen/duo/full-text pipeline.

The registered CPU controls in `scripts/analysis/stress_fourth_vote.py` add a
duplicate or label-independent hash ordering to the current fusion, without
loading a fourth model. All values below use 429 development questions:

| Fourth vote, weight 1 | Recall | Delta vs three voters | Better / worse |
| --- | ---: | ---: | ---: |
| No fourth vote | 0.952602953 | 0 | 0 / 0 |
| Copy of V9 | 0.952020202 | -0.000582751 | 0 / 1 |
| Copy of short duo | 0.951437451 | -0.001165501 | 1 / 2 |
| Copy of full-text duo | 0.949106449 | -0.003496503 | 0 / 2 |
| Uninformative hash ordering | 0.883644134 | -0.068958819 | 1 / 35 |

These are sensitivity controls, **not predictions of ViRanker performance**.
Duplicating a good component supplies no new information and can still lose
gold through reweighting. Lower weight reduces influence but does not eliminate
risk: a 0.25-weight short-duo copy loses the entire +0.002331 gain. We did not
select a deployment weight from these controls. No fourth-voter parameter
search should omit the no-fourth-voter baseline.

A real fourth-voter trial should first benchmark a small fixed question sample
for cost, then score development only. Keep candidates and the other voters
frozen. Report new gold retained, gold displaced, wins/losses and set changes
against both three-voter fusion and full-text duo. If tuning its weight, select
it only inside fitting folds, including an off option, and then lock it before
separate evaluation. Promote only after the required pooled +0.0052 gate; a
single development win is insufficient. These historically reused sets still
do not provide a fresh private-performance guarantee.

Sensitivity report: `reports/current-fusion/20260921-fourth-sensitivity/report.json`.

The historical `v8-partner_viranker.run.json` recorded 567,755,777 parameters.
Combined with the current 2,327,063,553 parameter audit, that projects:

| Budget item | Parameters |
| --- | ---: |
| Existing embedding, two rerankers and shared comparator | 2,327,063,553 |
| ViRanker, historical loaded count | 567,755,777 |
| Combined | **2,894,819,330** |
| Remaining below 4B | **1,105,180,670** |

The GPU scorer verifies downloaded file hashes and immutable commit provenance,
requires the exact ViRanker repository, enforces BF16 without quantization and
checks the actual loaded parameter count. Short/full-text duo are counted once
because they share the comparator checkpoint. It rejects a total at or above 4B.

For each candidate it obtains the top two document-local full-text BM25 windows,
with corpus-prefix windows only when no term matches exist. It scores each
query/passage pair at 512 tokens, combines the highest logit plus 0.2 times
the second, and exports a complete fourth ordering. No gold label affects
candidate choice, passage choice, scoring or ranking. Gold is checked only to
restrict this entry point to matching labeled inputs. Null-label private inputs
are rejected. Query/checkpoint/evidence identities bind resumable per-question
files; a complete resume does not load model weights.

`scripts/analysis/evaluate_fourth_voter.py` adds that ordering at weight 1 to
the fixed three-voter RRF and reports the result against both full-text duo
and three-voter fusion. It does not pick weights or produce a submission.
Both labeled splits are still required before a pooled promotion decision.

**Not run on a GPU yet:** no new ViRanker scores, fourth-voter accuracy or runtime
have been measured. Ten CPU tests passed, including a fake-model integration
test for scoring, complete resume without model reload and stale-cache rejection.

## Reproduction and future labeled scoring

CPU, from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/validate_rank_fusion_probe.py
.\.venv\Scripts\python.exe scripts/analysis/evaluate_current_fusion.py
.\.venv\Scripts\python.exe -m unittest tests2.test_current_fusion tests2.test_fourth_voter tests2.test_labeled_gate
```

The following Linux commands describe the optional labeled scoring stage after
restoring the frozen store, FTS index, baseline audit, dataset and exported
rankings at the shown relative paths. They are not a complete fresh-instance
restore installer. Use the existing verified model downloader first:

```bash
python -m legalir2 download --which partner_viranker
python -m legalir2.fourth_voter \
  --rankings reports/current-fusion/20260921-current-v2/development-rankings.json \
  --questions reports/private/fulltext-prior-release-check/dataset/development.json \
  --model models2/partner_viranker \
  --base-audit analysis/impact/duo-20260918T110105Z/audit.json \
  --index artifacts2/frozen-20260918 \
  --evidence-index artifacts2/frozen-20260918/duo-fulltext-v2.sqlite \
  --out outputs/fourth-voter/development-v1
python scripts/analysis/evaluate_fourth_voter.py \
  --rankings reports/current-fusion/20260921-current-v2/development-rankings.json \
  --questions reports/private/fulltext-prior-release-check/dataset/development.json \
  --fourth-run outputs/fourth-voter/development-v1 \
  --split development --out reports/current-fusion/fourth-development-v1
```

Use a new output directory if inputs, code, evidence or batch size change.
The existing +0.0052 pooled score requirement remains in force. The earlier
+0.008 FTS coverage prerequisite concerned adding new retrieved candidates;
this experiment measures voting over already available candidates separately.
