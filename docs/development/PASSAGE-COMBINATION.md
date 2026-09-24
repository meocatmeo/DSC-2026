# CPU combination after passage-order regression

The user reported 0.9529 for the primary passage-order ZIP, below the preserved
0.97 agreement winner at 0.9533. The primary changed 500 answer sets and removed
at least one previous top-four document from the final five on 89 questions.
The user requested a CPU-only combination before spending remaining attempts.

## Fixed policy

`legalir2/passage_combination.py` first reproduces the original 0.97 winner from
the original-evidence comparator scores. Its first four answers are frozen.
An outsider can replace only the winner's actual fifth answer when:

- Both passage-order comparator preferences against that fifth are at least 0.97.
- Both original-evidence comparator preferences against it are strictly above 0.5.
- The challenger has positive fitted ability advantage in all four score rows.

Choose by minimum new preference descending, minimum four-row ability margin
descending, then document ID ascending. Otherwise return the winner unchanged.
There is no refitting, threshold sweep, new retrieval or GPU inference. These
are two checkpoints under two evidence orders, not four independent voters;
pair preferences are not calibrated probabilities of relevance.

The policy was frozen before its development evaluation. The initial evaluation
attempt stopped before reading labels because it looked for prepared questions
in the completed-score folder. The corrected path uses the original verified
prepared folder; the policy did not change. The superseded plan is retained as
`reports/passage-combination-v1/plan-before-input-path-fix.json`.

## Development evaluation

All 429 questions were evaluated against the current 0.97 winner. Recall stays
0.9526029526, with seven changed sets, zero wins, zero losses, and paired
bootstrap interval [0, 0]. This reused development set supplies no measured
improvement or independent promotion evidence. No 1,050-question validation
evaluation is claimed. Thirteen targeted policy and anchor tests passed.

## Provenance and reproduction

```powershell
.venv\Scripts\python.exe -m unittest tests2.test_passage_combination tests2.test_agreement_097 tests2.test_duo_agreement
.venv\Scripts\python.exe scripts/analysis/combine_passage_scores.py evaluate
.venv\Scripts\python.exe scripts/analysis/combine_passage_scores.py package
```

The script checks pinned development/private plan digests, all score-file hashes
against completed GPU receipts, unchanged-input replay receipts, exact question
and prior alignment, prepared-input checksums, preserved edges and fitted-score
integrity. Private packaging requires exact replay of both the 0.9533 winner and
the passage-order ZIP before combining their saved scores. It verifies complete
question coverage, five unique known document IDs and exact first-four retention.

A development regression blocks packaging. A tie permits an explicitly
experimental ZIP for inspection, without promotion or competition submission.
If the candidate duplicates an existing ready ZIP, no new ZIP is packaged.
Reports are under `reports/passage-combination-v1/`; private artifacts are under
`submissions/private/passage-combination-v1/`. The ready filename, when a distinct
candidate exists, is `submissions/ready/submission-test-passage-combination.zip`.
Verified compact score records are saved for CPU recovery. Private labels remain
unavailable; none are inferred from aggregate leaderboard scores.

## Packaged private candidate

The verified private run changes **43 fifth answers** relative to the 0.9533
winner, retaining the original first four for all 2,080 questions. It differs
from every existing ready submission. Both the old winner and the 0.9529
passage-order predictions replayed exactly from their source scores before
combination. All 4,160 comparator score files matched the completed GPU receipt.

Ready ZIP: `submissions/ready/submission-test-passage-combination.zip`.
SHA256: `c179e33fddeab08f0a8d9cd11d8657845f3d3cbd4415df9e3ba4cc15e9d0f4bb`.
Compact recovery scores SHA256:
`ed84ae72ab7839924d5d395c146851dea90d02aa09739e2a25249a74b14d6eff`.

This is a limited-change exploratory candidate. Seven neutral development
replacements supply no measured improvement; the private score is unknown.
No competition submission was sent, and the preserved 0.9533 winner was not
replaced. Packaging consumed no additional GPU inference or submission attempt.

## Reported private result: 0.9538

The user subsequently reported **0.9538**, a displayed gain of +0.0005 over the
previous 0.9533 winner. Promote this exact ZIP as the current private reference;
the earlier development result remains a tie and must not be rewritten as a win.
The aggregate private score does not identify which of the 43 substitutions
helped or hurt. The score is user-reported, not independently retrieved.

Ready winner: `submissions/ready/submission-best-0.9538.zip` (same SHA256 as above).
Frozen handoff: `releases/handoff/20260923-best-09538-v1`. Standalone CPU recovery
recomputes all 2,080 predictions and reproduces the scored ZIP byte-for-byte.
This verifies saved-score recovery, not fresh model inference or a new Docker
image. The old tested Docker image still reproduces 0.9523.

## Final fixed extension at 0.95

The user requested a candidate for the final submission after reporting 0.9538.
`legalir2/passage_combination_final.py` lowers only the new-evidence preference
minimum from 0.97 to 0.95, retaining the original-evidence preference floor and
all four positive ability margins. It still starts from the original 0.97
agreement anchor, not recursively from the combined answers. The higher
confidence existing rescues remain highest-ranked; the generator additionally
checks that all 43 previously accepted replacements are unchanged.

`scripts/analysis/final_passage_combination.py` freezes this single policy before
evaluation. Relative to the 0.9538 algorithm, development has 11 changed sets,
zero wins/losses, and unchanged recall 0.9526029526. Sixteen targeted tests pass.
This is reused development and supplies no demonstrated improvement.

The private candidate changes 49 additional fifth answers, preserves the first
four for all 2,080 questions and retains all 43 prior combination replacements.
It is distinct from existing ready ZIPs. Private score is unknown; no competition
submission was sent and no GPU rerun occurred. Keep 0.9538 as the current winner.

Ready file: `submissions/ready/submission-final-passage-combination-095.zip`.
SHA256: `1436424c7d830d73f3056a6262d1f4796529907702cc9db4b2bc26503205d65f`.
Plan/review/changes are under `submissions/private/passage-combination-final-095-v1`.
Recovery scores are reused from the immutable 0.9538 handoff after hash and
exact winner replay checks. This is an exploratory final attempt, not a safer
or proven improvement over the existing best.

The user subsequently reported **0.9533** for the final trial, a displayed loss
of 0.0005 relative to 0.9538. Retain the original CPU combination at 0.9538 as
the final reference. Do not promote the 0.95 extension or overwrite the frozen
0.9538 source, scores, ZIP or recovery snapshot. The aggregate score does not
identify which of the 49 additional substitutions caused the loss.
