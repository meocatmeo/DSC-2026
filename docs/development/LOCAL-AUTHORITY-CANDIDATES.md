# Supporting-authority retrieval: CPU experiment

22 September 2026. The best user-reported private result remains **0.9523**
from the fifth-slot agreement rule. This experiment creates candidate proposals,
not new predictions. It does not modify that submission or its policy.

## Hypothesis and implementation

The CPU miss audit found supporting statutes and regulations outside the duo
pool. The earlier citation graph recognizes document numbers containing a year;
it cannot recognize yearless letters or a law cited by name alone.

`legalir2/local_authorities.py` adds those identity channels. Document-number
ownership comes from the dedicated corpus metadata field. Named laws come from
titles, with ambiguous names excluded unless an explicit version resolves them.
An unknown adjacent year never falls back to a different known version. Titles
of amendments do not claim ownership of the original law. Issuer suffixes such
as QH15 retain their digits.

The policy reads references from the first four frozen answer documents. It
proposes at most four documents outside the existing pool, using query overlap
around the reference and support from different source documents. Repetition
inside one source does not increase support. Gold labels are used only after
proposal generation, to report coverage. No corpus IDs or question IDs are
hard-coded in the rule.

The scorer and final answer ranking have not been changed. A citation is a
retrieval clue, not proof that the cited document answers the question.

## Development measurements

All measurements use the same historically inspected **429 development
questions**, with verified saved score digests and frozen corpus metadata.
These are not independent holdout results. The 1,050 validation questions and
private questions were not evaluated.

| Reference text | Gold instances recovered | Candidate additions | Pool coverage | Coverage delta |
| --- | ---: | ---: | ---: | ---: |
| Existing duo pool | — | — | 0.96425796 | — |
| Selected answer passages | 1 | 279 | 0.96542347 | +0.00116550 |
| All stored chunks of the four anchors | 2 | 1,635 | 0.96775447 | +0.00349650 |
| Original source text of the four anchors | 2 | 1,635 | 0.96775447 | +0.00349650 |

Selected passages recover document 280768 for question 2706. Stored chunks
also recover document 216530 for question 132380. Neither the new named-law
channel nor yearless numbers has demonstrated an additional recovery in these
three checks.

Removing the four-candidate cap does not improve the selected-passage coverage.
For stored chunks, the unbounded link ceiling is 0.97008547, an increase of
0.00582751 over the original pool. This is below the requested +0.008 coverage
prerequisite even before ranking losses. It is not an upper bound on other
retrieval methods or on the private score.

The legacy chunker can omit text between its short header and the first parsed
article. The evaluator also supports `--scope raw`, checking the original
corpus byte digest and document membership before evaluating full source text.
The raw-source check produced the same coverage and two recovered gold
instances as the stored-chunk check. Its unbounded ceiling is also 0.97008547.
The missing pre-article text does not unlock another gold recovery under this
reference-matching policy on these 429 questions.

## Decision

Do not activate GPU scoring or replace the best submission from this result.
Even the measured unlimited-link opportunity, +0.00582751, misses the user's
+0.008 candidate-coverage prerequisite. The four-candidate rule adds 1,635
candidate occurrences to recover only two gold instances; whether either would
survive comparator scoring is unknown. None of these figures estimates a
private leaderboard increase.

Further work needs a complementary candidate signal, rather than a larger cap
on this channel. Named laws and yearless numbers remain useful parser coverage,
but this experiment has not demonstrated a recall benefit attributable to them.
The next ranking experiment should retain the first-four protection and compare
new candidates against the displaced answer, avoiding whole-pool reranking.

## Reproduction and artifacts

From the repository root, with the verified local caches and frozen corpus:

```powershell
.venv\Scripts\python.exe -m unittest tests2.test_local_authorities tests2.test_duo_agreement
.venv\Scripts\python.exe scripts/analysis/evaluate_local_authorities.py --scope passages
.venv\Scripts\python.exe scripts/analysis/evaluate_local_authorities.py --scope documents
.venv\Scripts\python.exe scripts/analysis/evaluate_local_authorities.py --scope raw
```

Current evaluator defaults to `reports/local-authorities-v3-<scope>`; `--out`
selects a new experiment directory. A changed policy or evaluator is refused
when an existing frozen plan differs. Reports and corpus assets are ignored by
Git and must be supplied separately.

The passage and stored-chunk measurements above are preserved in
`reports/local-authorities-v2-passages` and
`reports/local-authorities-v2-documents`, including the evaluator snapshot used
for those runs. The complete-source measurement is in
`reports/local-authorities-v3-raw`, with its evaluator and module snapshots.
The earlier v1 report is retained as a historical diagnostic;
v2 fixes explicit-year matching and digit-bearing issuers. These were parser
correctness fixes, not threshold choices based on private feedback.

Each run writes a frozen plan, per-question candidate proposals with source
contexts, and a coverage report. No GPU release or submission is produced.
Thirteen unit tests passed across the new authority retriever and the winning
agreement rule. The best ZIP checksum still matches the recorded 0.9523 artifact.
