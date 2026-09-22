# CPU audit: retrained duo 0.9500 versus full-text baseline 0.9510

22 September 2026. Keep the 0.9510 baseline. The retrained comparator changed
many answers but did not address most measured remaining misses. No corrupt
submission or mismatched candidate cache was found. Private gold labels are
unavailable, so the particular private questions responsible for the decrease
cannot be identified from these scores.

## Scope and artifact identity

This audit replays all 2,080 private answers and the **already-computed** 429
development comparisons on CPU. It does not run the 1,050-question validation,
train a model, call GPU inference, alter a submission, or tune a new policy.
Eight private changes were chosen deterministically for passage inspection:
two largest changes, two strongest reversals of baseline score preference,
two smallest new boundary margins, and two SHA256-ordered control cases.
This sample is diagnostic, not an estimate of private accuracy.

- Baseline ZIP SHA256:
  `40caa611cfdbf4f853296fdfef93774b117fffc4c339d591690080a59e0a6d57`.
- Candidate ZIP SHA256:
  `1693ed68fd7d60a672875f4ee62c26c54e87f2504a25a7cddcd66bb40e4c3214`.
- Candidate run: `duo-fulltext-retrain-private-control-4090-v1`.
- Scores 0.9510 and 0.9500 are user-reported; the latter is associated with the
  completed candidate run in this conversation, not an independently downloaded
  leaderboard record.

Both ZIPs have exactly the expected questions and reproduce exactly from their
cached rankings. Candidate rows have identical candidate IDs, prior, question
and gold placeholders to their source rows; every source-row digest matches.
All 208 required baseline shard hashes match the frozen reference manifest.
All candidate score vectors/matrices and fitted gradients pass verification.
The recorded development score digest reproduces exactly, including both arms.

Rclone reported duplicate source objects for five baseline shard names. The
downloaded copies passed the frozen SHA256 checks; no cloud files were moved,
deduplicated or changed. Duplicate storage is an operations issue, not evidence
of incorrect predictions in these verified files.

## What changed

| Measure | Private, 2,080 questions | Development, 429 questions |
|---|---:|---:|
| Changed top-five sets | 719 (34.6%) | 130 (30.3%) |
| Changed order only | 593 | 124 |
| Replaced document slots | 747 | 139 |
| One / two / three replacements | 694 / 22 / 3 | 121 / 9 / 0 |
| Labeled recall wins / losses | Unknown | 1 / 0 |
| Changed sets with unchanged labeled recall | Unknown | 129 |

For 573 of the 719 changed private questions, only the baseline's fifth-ranked
document was removed. On 436 questions the deepest added document had been
sixth. Changes therefore concentrate near the cutoff, although 21 questions
also remove a document previously ranked second or third.

The new model is **less**, not more, extreme overall: the fraction of pair
probabilities outside [0.1, 0.9] falls from 44.93% to 41.62%; average binary
entropy rises from 0.533 to 0.561 bits. This is descriptive, not a calibration
assessment. The same pattern appears on development. An explanation based on
the new model simply becoming overconfident is not supported by this measure.

In 241 of 809 added-versus-removed comparisons, the new model's direct pair
preference favors the removed document, despite the final ranking admitting
the other one. These are cross-products for multi-document changes, not 809
independent questions. Bradley–Terry fitting uses the whole comparison graph
and a prior, so this is not itself a bug. It shows why a final-rank flip is not
the same as a direct, well-supported replacement. A direct-pair safety rule
would need labeled testing; its private benefit cannot be inferred here.

## Why the measured improvement was small

The only development improvement is question **162906**, the IELTS threshold
for exemption from the 2022 school-leaving foreign-language exam. Gold document
**120889**, circular 1523/BGDĐT-QLCL, moves from seventh to third. The other 129
changed sets recover no new labeled documents and remove none.

The rebuilt-evidence model has 403/429 questions with full labeled recall.
Across the remaining 26 questions it misses 31 query-document gold pairs:

- **24 are outside the comparator pool** across 20 questions. Five are already
  in the saved V9 candidate list (positions 18, 26, 29, 31 and 46), but were not
  admitted to the smaller comparator pool. Nineteen are absent from the saved
  V9 list. This does not establish absence from the corpus or all raw retrievers.
- **Seven are inside the comparator pool** across seven questions. One question
  has both kinds of miss, hence 20+7 is greater than 26.

Current candidate coverage has macro recall **0.964258** on development versus
actual **0.952603**. Perfect ranking within this exact pool would add at most
0.011655 on this split. That is a labeled ceiling, not an expected private gain.
The 24/31 count is document-pair weighted; it must not be confused with macro
recall weighting.

The seven ranking misses do not have one common remedy:

| Question | Gold ID | Prior rank | Fitted ability rank | Final rank | Observation |
|---|---|---:|---:|---:|---|
| 25436 | 160381 | 18 | 4 | 17 | Tax-exemption procedure document remains suppressed by weak prior |
| 6674 | 6986 | 17 | 6 | 17 | Exam-grading phone question; displayed evidence starts with regrading logistics |
| 19560 | 27980 | 14 | 14 | 14 | Administrative-unit classification source remains low in both signals |
| 114846 | 135923 | 12 | 4 | 7 | Comparator promotes the National Assembly law, but not enough for top five |
| 124546 | 285503 | 17 | 6 | 14 | Visible customs passage explicitly enumerates non-collection cases |
| 162020 | 247495 | 4 | 17 | 8 | Comparator demotes the labeled corporate-tax source; first visible passage concerns wages rather than charitable deductions |
| 88726 | 160381 | 17 | 2 | 13 | Visible passage contains the requested tax-installment procedure; prior still dominates |

Abilities themselves are regularized by the prior. Final scores then use
`0.75 * prior + 0.25 * ability`, so the prior affects both fitting and blending.
This is the existing deliberate policy, not newly introduced corruption.
Question 88726 strongly illustrates its cost: the new pair probability favors
the gold document over the selected fifth document by 0.99925, yet its final
rank is 13. Conversely, simply strengthening this comparator can worsen cases
like 162020. Do not infer that globally increasing its weight is a solution.

## Bounded manual passage review

Observations below concern supplied corpus passages, not current legal advice
or verified hidden answers. A questionable added document does not establish a
recall loss if the gold document remains elsewhere in the top five.

| Question | Changed documents and inspected evidence | Diagnostic lesson |
|---|---|---|
| 144638 | A named mountain-school charity question swaps three other-fund/general-fund documents for three different other-fund/general-fund documents; first two answers stay | Generic organization wording can create large, potentially irrelevant answer churn |
| 60202 | Correcting pedagogical certificates swaps cross-agency certificate-management documents for broader pedagogical training and college provisions; first two stay | Exact certificate type, issuing authority and requested action need joint matching |
| 142292 | Added 174679 matches the named labor confederation but shows organizational duties; removed 209839 describes awards procedures for Hanoi | Entity match alone and action match alone each have scope risks; neither change is proven correct |
| 63170 | Added 8623 contains classification levels for subordinate units; query asks about audit-team members. Removed 193192 concerns audit-file management | Matching an answer-shaped list can still select the wrong subject |
| 104904 | Added 164447 contains paternity-leave wording but its label explicitly says draft; removed 208105 has broader employment/leave passages | Version/status is a useful diagnostic dimension, but automatic exclusion of drafts is unsupported without dataset labels |
| 138474 | Primary/secondary-school defense education replaces an annual defense-education plan for a question about course principles | Educational stage and document purpose matter beyond topic overlap |
| 57514 | General penalty law replaces maritime penalties for an aviation working-hours query; first four answers stay | Both inspected changed candidates lack the requested aviation-specific rule; reverting is not demonstrably better |
| 286 | Added finance/security-cooperation circular mentions financial forensic expertise; removed law amendment explicitly discusses forensic case-file composition | Topic phrase matching may outrank the specific requested procedural evidence |

## Next work justified by this audit

1. **Audit the 24 development admission/retrieval misses on CPU first.** Separate
   the five available V9 candidates from the nineteen missing candidates. Inspect
   exact document identifiers, cited source relationships and packed passages.
   The existing 72-policy FTS sweep already failed its +0.008 prerequisite;
   repeating the same threshold sweep is not a new retrieval strategy.
2. **Study evidence-backed promotion for weak-prior documents**, using labeled
   data: require support for the requested entity, action and scope, plus direct
   support against the displaced fifth document. Keep the baseline otherwise.
   This is a proposed general rule, not an implemented or measured improvement.
3. **Inspect passage selection before another comparator retrain.** The selected
   evidence for some known gold documents is about adjacent procedures, while
   others already show the exact rule but lose through the prior. These require
   different fixes. Do not bundle passage changes, new negatives and new blending
   weights into one unidentifiable experiment again.

No new submission was created. No private answers were manually relabeled or
patched. No policy was selected from private feedback. Further labeled work
must acknowledge historical reuse of these splits; this audit is diagnostic,
not independent confirmation of an improvement.

## Reproduce and inspect

Run `.venv/Scripts/python.exe scripts/analysis/audit_retrained_private.py` after
restoring the same artifacts. The script checks frozen hashes, cached fitted
scores, exact ZIP replay, shared priors/pools, and the development score digest.
Reports live under `reports/retrain-private-audit/` (ignored data artifacts):

- `summary.json`, `source-hashes.json`: measurements and provenance.
- `changes.json`, `development-changes.json`: full mechanical comparisons.
- `labeled-in-pool-misses.json`, `labeled-admission-misses.json`: bounded labeled
  follow-up lists and saved passage/rank evidence.
- `sample.json`, `sample-passages.txt`: eight-case sample and complete packed
  evidence, selected by the deterministic rules above.

The script reads only the DSC artifacts downloaded for this audit. It does not
write rclone, touch AIC/NCKH, or change the scoring pipeline.
