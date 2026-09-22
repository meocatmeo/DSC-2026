# Deadline submission alternatives

22 September 2026. The user has four submission attempts and one day remaining
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
Assuming only this one attempt was used, three attempts remain.
