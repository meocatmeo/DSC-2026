# CPU FTS coverage sweep and admission gate

Measured 21 September 2026. **Coverage failed the requested +0.008 gate.**
Keep the private full-text duo reference scoring 0.9510. No new inference,
submission or GPU run was performed, and no production ranking was changed.

## Measured result

The fixed grid contains 72 policies: contiguous phrases of 3/4/5/6 words,
query-term coverage of 0.25/0.40/0.55, 2/4/8 extra documents, and the first
96/256 FTS passage hits. Extra documents are appended after preserving the
full-text baseline candidate pool, including its existing phrase additions.

Best pooled policy: **5 words, 0.25 term coverage, 8 extra documents, 96 hits**.

| Labeled set | Questions | Baseline pool coverage | Expanded coverage | Change |
| --- | ---: | ---: | ---: | ---: |
| Development | 429 | 0.964258 | 0.966589 | +0.002331 |
| Validation | 1050 | 0.975714 | 0.979048 | +0.003333 |
| Pooled | 1479 | 0.972391 | 0.975434 | **+0.003043** |

This recovered two gold documents on development and four on validation,
across six questions. Recall gives each question equal weight, so the gain
corresponds to 4.5 full-question equivalents, not six. The required +0.008
would exceed 11.832 equivalents. No policy in the grid cleared that threshold.

Coverage means the fraction of a question's gold documents present anywhere
in the candidate pool, averaged across questions. It is not final top-five
recall. The approximate 96% ranking conversion is a planning assumption; the
ratio of old recall to old coverage does not establish a 96% success rate for
newly retrieved documents.

## What was frozen and verified

- Corpus: 8,532 documents and 467,589 full-text windows. The SQLite FTS index
  and source store were opened read-only; their hashes and metadata were checked.
- Development uses the saved full-text baseline pool and verified score cache.
- Validation uses the frozen original duo pool, then reconstructs the existing
  full-text-v2 phrase additions on CPU. This establishes candidate coverage,
  **not full-text validation answer predictions or final recall**.
- Dataset membership, question text, gold labels, source document IDs and cache
  identities are checked. Private input is never loaded by the sweep.
- Policies and code/data hashes are written before measurement to
  `registered-plan.json`. Both labeled sets have been used historically; this
  is an exploratory pooled sweep, not a fresh holdout evaluation.

To avoid unnecessary work, queries whose gold is already fully covered skip
FTS search. This label-aware shortcut is valid only for measuring coverage:
their before/after coverage is necessarily 1. It must never become inference
logic. Fifty-nine uncovered queries were searched. The `diagnostic_additions`
field counts only these diagnostic queries, not inference-wide additions or
cost. Query search and measurement took about 175 seconds, excluding startup
hash verification and cache loading.

## Conditional next step and score gate

The requested append-only admission experiment is **not activated**, because
its coverage prerequisite failed. If a future retrieval proposal clears it,
freeze the baseline top five and admit a new document only when its measured
comparison clears a registered margin against the specific displaced member.
Do not recompute the baseline ordering or globally rerank the enlarged pool.
This limits churn; it does not guarantee better accuracy. The earlier focused
rescue already tried bounded, confident replacements and still regressed.

`legalir2/labeled_gate.py` implements the requested pooled score deadband:

- Delta **>= +0.0052**: promote under this numerical rule.
- Delta **<= -0.0052**: reject.
- Otherwise: inconclusive; keep the baseline.

Delta is the mean of per-question proposed recall minus baseline recall over
all 1,479 labeled questions, not an unweighted mean of the two split means.
The gate validates complete paired predictions, nonempty gold and five unique
known document IDs, and reports split scores, wins, losses and changed sets.
Exact rational arithmetic avoids float rounding at the threshold boundary.
This deadband is **not a confidence interval or a guarantee of private gain**.

`scripts/analysis/gate_fts_admission.py` first refuses evaluation when the
coverage gate fails. It then requires complete actual paired predictions with
the matching coverage plan, policy and full-text method identity. Source cache
verification remains the responsibility of the prediction exporter: metadata
alone cannot prove that neural inference was performed. No valid prediction
bundle currently exists for this experiment, and the score gate has not passed.
This is an offline gate; legacy GPU/submission scripts are unchanged.

The future input bundle format is:

```json
{
  "coverage_plan": "digest of registered-plan.json using legalir2.h100.digest",
  "baseline_method": "duo-fulltext-v2",
  "baseline_identity": "verified baseline inference identity",
  "proposal_identity": "verified proposal inference identity",
  "policy": {"phrase_words": 5, "term_coverage": 0.25, "extra": 8, "search_limit": 96},
  "splits": {
    "development": {"question-id": {"baseline": ["a", "b", "c", "d", "e"], "proposed": ["a", "b", "c", "d", "f"]}},
    "validation": {"question-id": {"baseline": ["a", "b", "c", "d", "e"], "proposed": ["a", "b", "c", "d", "f"]}}
  }
}
```

The IDs above are illustrative; both splits must contain every real question.

## Reproduction

From the repository root with existing local frozen assets:

```powershell
.\.venv\Scripts\python.exe -m unittest tests2.test_fts_sweep tests2.test_labeled_gate
.\.venv\Scripts\python.exe scripts/analysis/sweep_fts_coverage.py
```

The six tests cover FTS quoting, equivalence to frozen candidate expansion,
read-only access, document exclusions/deduplication, search caps and thresholds,
paired macro-recall pooling, exact gate boundaries and invalid/incomplete data.

The resumable report directory is `reports/private/20260921-fts-coverage/`:
`registered-plan.json`, `inputs.json`, per-query search caches and `report.json`.
Changed code or inputs require a new `--out` directory. Reports and data remain
gitignored; the two entry points and their existing cache-loader dependencies
are explicitly exempted so the source can be reviewed and shared through Git.
All `.sh` files remain ignored.

For a future proposal that passes coverage, the offline score check is:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/gate_fts_admission.py --predictions <paired-predictions.json> --coverage <coverage-report-directory> --out <new-gate-report-directory>
```

The current coverage directory will intentionally fail this command. It does
not authorize a submission, launch a GPU or alter a ZIP.
