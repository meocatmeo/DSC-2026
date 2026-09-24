# Deadline submission alternatives

Historical setup, 22 September 2026. The user had four submission attempts and one day remaining
and explicitly accepts speculative attempts. These CPU alternatives therefore
do not require the earlier +0.008 candidate-coverage prerequisite. None has a
measured private score. The current best remains the reported **0.9523**.

## Fixed experiments

The three settings were recorded before development evaluation. There is no
threshold grid search or tuning to individual private questions. The original
0.9510 rankings and the two saved comparator matrices remain the inputs.

| Candidate | Change relative to the 0.9523 rule |
| --- | --- |
| `fifth-wide` | Lower both-model pair preference from 0.90 to 0.80; permit extra fifth-slot replacements |
| `fifth-strict` | Raise it to 0.98; revert less strongly supported fifth-slot replacements |
| `fourth-rescue` | Preserve the current fifth answer; allow one outsider to replace the original fourth when both models prefer it by at least 0.95 |

All replacements also require positive ability advantage in both models.
Challenger choice uses the same minimum pair preference, minimum ability margin,
then document-ID ordering as the existing agreement rule. No ability is refit.
These model preferences are not calibrated probabilities of answer correctness.

Fifth variants protect the first four answers. Fourth rescue protects the first
three and the current fifth, and cannot reinsert any original top-five document
as its outsider. Each candidate changes at most one answer per question relative
to the current best. The fourth variant can differ in two slots from the old
0.9510 baseline because it retains the existing fifth-slot improvement.

## Development results

Reference: current agreement, recall **0.9526029526** on 429 development questions.

| Candidate | Recall | Wins / losses | Changed answer sets | Paired 95% interval for delta |
| --- | ---: | ---: | ---: | --- |
| Fifth wide | 0.9502719503 | 0 / 1 | 42 | [-0.006993, 0] |
| Fifth strict | 0.9526029526 | 0 / 0 | 49 | [0, 0] |
| Fourth rescue | 0.9526029526 | 0 / 0 | 23 | [0, 0] |

These are exploratory comparisons on historically reused development questions.
Intervals are descriptive and not adjusted for selecting among three variants.
Zero observed recall differences do not establish equivalence or safety on the
private test. Private labels remain unavailable. No 1,050-question validation
run or GPU inference was performed.

## Artifacts and reproduction

Code: `legalir2/deadline_agreement.py`. Tests:
`tests2/test_deadline_agreement.py` and `tests2/test_duo_agreement.py` (8 passed).

```powershell
.venv\Scripts\python.exe -m unittest tests2.test_deadline_agreement tests2.test_duo_agreement
.venv\Scripts\python.exe scripts/analysis/deadline_agreement.py evaluate
.venv\Scripts\python.exe scripts/analysis/deadline_agreement.py package
```

The frozen plan and labeled results are under `reports/deadline-agreement-v1`.
Separate candidate directories are under
`submissions/private/20260922-deadline-agreement-v1`. Each contains a ZIP,
checksum, per-question changes, and a review recording the measured development
result and private answer-change count. Reports, caches and submissions are
gitignored. A code-only checkout is insufficient to reproduce the ZIPs.

Packaging verifies original shard hashes, comparator input alignment and fitted
score residuals, exact replay of both source ZIPs and the 0.9523 ZIP, complete
membership of all 2,080 private questions, and five distinct known document IDs
per answer. Each resulting ZIP is reopened and compared against its predictions.
The script does not submit to the competition or alter the best ZIP.

Use one attempt at a time. The wider rule has an observed development loss and
is the least supported option. Preserve at least one of the four attempts rather
than submitting all alternatives blindly. Scores cannot identify which private
question-level edits won or lost.

## Packaged private alternatives

| Suggested order | Candidate | Changed sets vs 0.9523 | ZIP SHA256 |
| --- | --- | ---: | --- |
| First | Fourth rescue | 85 | `c49869057894cdcdf683d2041f063d7cc21caeb9dd0a506753498b662af7ec3c` |
| Second option | Fifth strict | 212 | `d8ea76dd918331fbd21067dfdaed31106d8f99de2d9b852e26c34792c389cae8` |
| Reserve, observed development loss | Fifth wide | 179 | `2272e6a076e68a598d8e0b86a0538c77a7d30e6724bdbde7e630a8f29471a8fa` |

Fourth rescue is suggested first because it retains the successful fifth-slot
policy and makes the fewest additional edits. This is a limited-change trial,
not a measured improvement or proof of lower private risk. All three ZIPs have
2,080 complete predictions. They were generated and checked locally, not
uploaded to the competition. No attempts have been consumed by this work.

Pairwise set differences: wide vs strict 391; wide vs fourth 247; strict vs
fourth 287. The alternatives are not three copies of the same predictions.

## First reported result

The user subsequently reported **0.9520**. This is attributed to fourth-rescue
from the preceding recommendation; the user did not restate the filename.
The change is **-0.0003** relative to the 0.9523 best, which remains intact.
Do not carry the fourth-slot changes into another candidate.

The next prebuilt option is fifth-strict, which protects all first-four answers
and reverses 212 of the existing fifth-slot replacements. Development tied the
current best rule. It may still regress; this score does not establish that a
stricter fifth threshold is better. Its policy and ZIP predate the new score.
Assuming only this one attempt was used, three attempts remained at that point.

## New best reported 23 September

The user submitted the provided fifth-strict ZIP and reported **0.9525**. This is
the new best, +0.0002 relative to 0.9523. Its SHA-256 is unchanged:
`d8ea76dd918331fbd21067dfdaed31106d8f99de2d9b852e26c34792c389cae8`.
The receipt is `fifth-strict/leaderboard-result.json`; the ready copy is
`submissions/ready/submission-best-0.9525.zip`. Earlier review reports retain
their pre-submission measurements rather than being rewritten as predictions.

This measured private gain does not show which of the 212 reversions helped or
establish that raising the threshold further will improve results. Preserve the
0.98 policy exactly. No private gold labels are available. Remaining attempt count
is not independently verified.

## One additional test requested after 0.9525

The user requested one more ZIP. The `fifth-099` policy raises the minimum weaker-
comparator preference from 0.98 to 0.99 and otherwise preserves the scored rule.
It restores the original fifth answer when a current rescue falls below 0.99;
all first-four answers remain identical. This is one fixed experiment, not a
threshold sweep, but it was proposed after seeing private leaderboard results.

Development versus the 0.9525 rule: 429 questions, recall 0.9526029526 unchanged,
eight changed sets, zero observed wins/losses. This does not establish private
equivalence or improvement. Private: 41 changed fifth answers; 2,080 valid records.
The 0.9525 ZIP is preserved and remains the best measured result.

Ready ZIP: `submissions/ready/submission-test-fifth-099.zip`.
SHA-256: `1a81b022f8be2005c975ee6e0fc5085f5ca4632d0d3b5b725c3cdb98bb39a00d`.
Code: `legalir2/strict_tail.py`; reproduction: `scripts/analysis/submit_strict_tail.py`.
Plan, development results and private changes: `submissions/private/20260923-fifth-099-v1`.
The generator replays the best 0.9525 answers from the immutable recovery scores
before producing the candidate. Twelve policy tests passed. No GPU inference,
private gold labels, or passage-order experiment outputs were used.

The user subsequently reported **0.9520 for fifth-099**, a loss of 0.0005 relative
to the best. Keep 0.98/0.9525. At the user's request, one fixed 0.97 alternative
was then packaged: 44 changed fifth slots, first four preserved; development tied
with six changed sets and zero observed wins/losses. Private score is unknown.
Ready ZIP: `submissions/ready/submission-test-fifth-097.zip`; SHA256
`34167af0e86aa5478992320319f319c45130d3ea894fb5b161deae661a4b4fc6`.
Code and generator are `legalir2/agreement_097.py` and
`scripts/analysis/submit_agreement_097.py`. This is adaptive exploration after
leaderboard feedback; the tie does not imply that 0.97 is safer or better.

The user then reported **0.9533 for fifth-097**, establishing the new best:
+0.0008 versus 0.9525. Both repeated messages report the same result, not two
separate experiments. The exact ZIP and source/cache recovery are preserved
under `releases/handoff/20260923-best-09533-v2` and
`submissions/ready/submission-best-0.9533.zip`. Recovery reproduces all ZIP bytes
and 138 retained fifth-slot replacements relative to the original 0.9510 anchor.

The v2 recovery snapshot passed its 226-entry release manifest and was marked
read-only. Cloud verification found 227 matching files and zero differences at
`gdrive:DSC-2026-results/releases/handoff/20260923-best-09533-v2`.
This preserves exact score-cache recovery; a new 0.9533 Docker image has not yet
been built and tested. The older frozen Docker image reproduces 0.9523.

The separate private passage-order release uses 0.97 as its primary policy and
also exports a 0.98 alternative from the same GPU comparisons. Neither output
has a measured private score. Its completed development experiment tied, so
the private run is user-requested exploration, not a promotion over 0.9533.

## Fixed 0.95 trial, 23 September

At the user's request, `legalir2/agreement_095.py` lowers only the minimum
pair preference to 0.95. Both comparators must still meet it and have a positive
ability margin; the first four answers stay fixed. The generator
`scripts/analysis/submit_agreement_095.py` verifies the frozen 0.9533 recovery
sources and replays every current-best answer before creating the trial.

The ZIP contains 2,080 questions and changes 76 fifth answers relative to the
0.97 winner. Reused development recall remains 0.9526029526 on 429 questions:
17 changed sets, zero wins and zero losses. Nine targeted tests passed. This is
one user-selected threshold following leaderboard feedback, not independent
evidence of improvement. Private score is unknown; 0.9533 remains the best.

Ready file: `submissions/ready/submission-test-fifth-095.zip`.
SHA256: `b08bee560e771049fb0f70d656549392611c5bf8d9e2f9785dfda3325df2b6e3`.
The trial reuses saved scores and needs no GPU rerun. The previously uploaded
passage-order GPU release remains immutable and separate from this trial.

The user subsequently reported **0.9533 for fifth-095**, tying fifth-097 at
the reported precision despite 76 changed answers. The aggregate result does
not distinguish neutral replacements from offsetting wins/losses or smaller
differences hidden by score rounding. Keep fifth-097 as the reference winner:
fifth-095 has no measured advantage. Record the result in the trial's
`leaderboard-result.json`; the pre-submission review remains historical.

## Requested 0.90 trial: duplicate of the earlier 0.9523 result

`scripts/analysis/submit_agreement_090.py` regenerated the original 0.90 rule
from the verified frozen scores. It changes 168 fifth answers from the 0.97
winner and ties reused development recall (43 changed sets, zero wins/losses).
All 2,080 prediction records, including document order, are identical to the
frozen original agreement ZIP reported at 0.9523. ZIP bytes differ, but parsed
predictions do not. No new competition submission was performed.

Ready copy: `submissions/ready/submission-test-fifth-090.zip`.
SHA256: `9db170d800779a665f75850fc3e5d8317865435ed2988f8146beee757a7af6ef`.
Evidence: `submissions/private/20260923-fifth-090-v1/duplicate-check.json`.
Recommend retaining the submission attempt: this is a reproduced previous
candidate, not an untested threshold. The 0.9533 reference winner is unchanged.

## Fixed 0.93 trial, 23 September

At the user's request, `legalir2/agreement_093.py` uses minimum pair preference
0.93 with the same both-model support, positive ability margins and protected
first four answers. `scripts/analysis/submit_agreement_093.py` verifies frozen
score sources and exact replay of the current 0.9533 winner before packaging.

The ZIP contains 2,080 verified predictions: 119 fifth answers change relative
to 0.97, 43 relative to 0.95, and 49 relative to 0.90. Parsed predictions differ
from every previous ZIP in `submissions/ready`; see the trial's
`duplicate-check.json`. Development recall remains 0.9526029526 on 429 questions,
with 32 changed sets and zero wins/losses relative to 0.97. Boundary and admission
safeguards were checked. This uses previously examined development and a
user-selected threshold following leaderboard feedback; private score is unknown.

Ready file: `submissions/ready/submission-test-fifth-093.zip`.
SHA256: `8b731f010909484a43a6f074dedf9136627973ead034e640463b490d54373222`.
No GPU rerun or competition submission was performed. The 0.9533 winner remains
preserved as the reference.

The user subsequently reported **0.9533 for fifth-093**. Thresholds 0.93,
0.95 and 0.97 therefore tie at displayed precision. Retain the already preserved
0.97/0.9533 release as the reference winner. These trials show no measured gain
from lowering the threshold in that range; they do not establish a global score
ceiling or identify which changed answers are correct. No further threshold
trial is supported by a measured improvement from this sequence.

| Minimum pair preference | Reported private score |
| --- | ---: |
| 0.90 | 0.9523 |
| 0.93 | 0.9533 |
| 0.95 | 0.9533 |
| 0.97 | 0.9533 |
| 0.98 | 0.9525 |
| 0.99 | 0.9520 |

## Private passage-order result

The completed GPU run produced a 0.97 primary ZIP with SHA256
`346cc59e9fdf57cbb58fb02d542ca4d74bd6bc94f40d0e6d084d941330f0c4c5`
and a 0.98 alternative with SHA256
`6d8b37b70fe602c4e6adb7a807ae0665c96a7da79293bb49498bdc7b7fc40c2a`.
Both contain 2,080 valid predictions and match the frozen-plan GPU receipt.
They change 500 and 495 answer sets respectively relative to the preserved
0.9533 winner and differ from each other on 44 questions.

The user reported **0.9529** immediately after the primary ZIP recommendation.
Attribute this result to the primary 0.97 ZIP unless corrected; the filename was
not restated. Its displayed delta is -0.0004. Keep the old 0.97/0.9533 winner;
do not promote the passage-order run. These new rankings do not freeze the old
first four: the primary drops at least one old top-four document from the final
five on 89 questions. Only the prefix within each rescored ranking is protected.

The 0.98 alternative remains unscored. Its 44 different answers could help or
hurt; the aggregate primary score cannot establish which. Do not automatically
spend another attempt on it. Two submissions are inferred to remain only if
this primary was the sole submission since the user reported three remaining.
Receipts and comparisons: `reports/passage-private-submission-review/`.

## CPU combination becomes the new best: 0.9538

After the passage-order regression, the fixed CPU combination retained the
0.9533 winner's first four and admitted only fifth-slot challenges supported by
both models under both evidence orders. It changed 43 private answer sets;
development tied with seven neutral replacements. The user then reported
**0.9538**, +0.0005 over the prior best. The new reference is
`submissions/ready/submission-best-0.9538.zip`, SHA256
`c179e33fddeab08f0a8d9cd11d8657845f3d3cbd4415df9e3ba4cc15e9d0f4bb`.

Source, frozen scores and exact standalone ZIP recovery are preserved at
`releases/handoff/20260923-best-09538-v1`. See `PASSAGE-COMBINATION.md` for the
policy and limitations. Do not infer which individual private answers improved
from the aggregate score or recursively apply the combination to its own output.
