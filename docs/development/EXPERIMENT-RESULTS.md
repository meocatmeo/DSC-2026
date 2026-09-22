# Experiment results and development lessons

Updated 22 September 2026. [Start here](README.md) for the current decision.

## Evidence levels

Private scores below are user-reported leaderboard results. We do not have
private labels or locally reproducible private recall. Development scores and
change counts were replayed from saved predictions and labeled data; the saved
score digests match. ZIP validation establishes format and provenance, not
accuracy. Candidate coverage is not final Recall@5.

## Score ledger

| Method | Development recall, 429 questions | Private score | Decision |
| --- | ---: | ---: | --- |
| Original V9 private | Different historical configuration | 0.9457 | Historical reference |
| Original short-passage duo | Different historical configuration | 0.9469 | Historical reference |
| Full-text duo baseline | 0.9502719503 | 0.9510 | Preserve as frozen ranking anchor |
| Deep fusion, 256 training questions | 0.9514374514 | 0.9464 | Private regression: -0.0046 |
| Deep fusion, 512 training questions | 0.9520202020 | 0.9459 | Private regression: -0.0051 |
| Retrained full-text comparator, matched control | 0.9526029526 | 0.9500 | Private regression: -0.0010; retain baseline |
| CPU agreement, fifth-slot replacement | 0.9526029526 | **0.9523** | New best reported private score; +0.0013 vs 0.9510 |
| Deadline fourth-slot rescue | 0.9526029526 | 0.9520 (attribution inferred) | -0.0003 vs best; retain 0.9523 |

The user reported 0.9520 after the recommendation to submit fourth-rescue first.
The filename was not restated, so that attribution is inferred from context.
Its 85 changed private answer sets did not improve the aggregate score. This
does not reveal which replacements were wrong. The stricter fifth-slot candidate
was already frozen and packaged before this result; it remains an unscored,
speculative next option, not an established improvement.

The user reported 0.9523 for the exact agreement ZIP on 22 September. Its SHA256
is `37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f`.
It changes 306 fifth-place answers and preserves all first-four answers relative
to 0.9510. This is an observed aggregate score improvement, not evidence that
every replacement is correct or that further threshold tuning will improve it.
The fitted models, fixed rule and original 0.9510 anchor are preserved.

The retrained control was submitted at the user's request without the remaining
validation. Its [CPU audit](RETRAINED-DUO-CPU-AUDIT.md) verifies all 2,080 cached
predictions and finds 719 changed private sets (747 replacements). Development
has one win and no losses among 130 changed sets; 129 changes are recall-neutral.
The rebuilt and old-evidence matched arms tied in development recall. The audit
separates remaining candidate-admission misses from ranking/prior failures; it
does not identify hidden private errors or establish a new improved policy.

The score message used "216"; mapping 0.9464 to the 256 model follows the user's
preceding explicit file identification and remains qualified pending correction.
Public 0.9508 and earlier validation metrics are different test/configuration
results and must not be mixed into this table as private measurements.

Increasing fusion training from 256 to 512 raised development recall by
0.0005827506 but lowered reported private recall by 0.0005. This does not prove
larger training sets always hurt; it does invalidate a monotonic-improvement
claim for these runs.

## Why the large answer changes were misleading

Verified development comparisons:

| Comparison | Changed answer sets | Better questions | Worse questions | Neutral changes |
| --- | ---: | ---: | ---: | ---: |
| 256 vs full-text baseline | 259 | 3 | 3 | 253 |
| 512 vs full-text baseline | 254 | 3 | 3 | 248 |
| 512 vs 256 | 83 | 1 | 0 | 82 |

The metric averages per-question gold recall. Equal counts of better and worse
questions need not cancel: the number of gold documents recovered or lost and
the number of gold documents per question affect the magnitude of each change.
"Neutral" means the number of gold documents retrieved stayed the same; it does
not establish that the substituted documents were equally relevant.

Verified private output comparisons against the scored 0.9510 ZIP:

| Candidate | Changed sets / 2,080 | One-document replacements | Order-only changes |
| --- | ---: | ---: | ---: |
| 256 | 1,223 (58.80%) | 994 | 557 |
| 512 | 1,219 (58.61%) | 1,004 | 545 |

The 512 and 256 outputs differ in answer membership for 370 questions: 369
replace one document and one replaces two. Another 400 differ only in order.
Order-only differences do not change set-based Recall@5.

Both uploaded archives contain exactly `submission.json`, cover all 2,080 IDs,
and contain five distinct known corpus IDs per question. Their SHA256 hashes
match the respective run sidecars. The regressions were not explained by an
incorrect ZIP being attached or an incomplete output file.

## Candidate coverage versus ranking

The completed 512 development run reported:

| Stage | Gold-document coverage |
| --- | ---: |
| Fused top 50 | 0.9665889666 |
| Fused top 100 | 0.9864024864 |
| Full scored pool, including preserved candidates | 0.9887334887 |
| Original duo pool | 0.9642579643 |
| Expanded duo pool | 0.9747474747 |
| Final five predictions | 0.9520202020 |

The measured gaps distinguish three issues: gold absent from scored candidates,
gold discarded before duo comparison, and gold present for duo but omitted from
the final five. They are diagnostic ceilings, not forecasts of attainable gains.
The 256 and 512 aggregate coverage values match; that does not establish that
their exact candidate lists match for every question.

An earlier frozen-V9 coverage audit used a slightly different retrieval path:
top 50 to top 100 exposed nine additional development gold-document instances
and ten on the 1,050-question validation set. Its development coverage was
0.967754 to 0.984071, not the current values above. Do not silently substitute
the old coverage metrics for the actual GPU-run metrics.

## Earlier rejected experiments

- **Full-text relevance prior and consensus:** the 19-policy development search
  selected the original prior (`alpha=0`, strength 0.25), recall 0.9502719503.
  The extra signal did not establish a development improvement.
- **Focused/citation rescue:** baseline remained 0.9502719503. Focused thresholds
  0.75/0.90 yielded 0.9415306915/0.9450271950; citation thresholds yielded
  0.9426961927/0.9461926962. Do not relaunch these as promoted policies.
- **CPU agreement/delta alternatives:** all tested variants regressed. A grouped
  inner-fold probe also regressed; see the [rescue audit](../state/AFTER-RESCUE-AUDIT.md).
- **Full 1,569-question deep-pool release:** packaged, but no result for that full
  training-size run is established here. The smaller pilots replaced that
  planned expense; do not invent a score or assume the full run occurred.

The teammate audit found no quantization implementation in the inspected
branches. The later inference runs explicitly used BF16. Quantization is not
a supported explanation for these measured regressions.

## Interpretation and next experiments

Established: the deep method found a larger candidate pool, changed many
predictions, gained slightly on reused development, and regressed privately.
The exact private errors are unknown. Overfitting to a small reused selection
set, distribution differences and poorly calibrated changes to the prior are
plausible explanations, not isolated causal findings.

The decision to promote 512 from one additional development win was too
optimistic. A correct implementation and a tiny development gain were treated
as stronger evidence than they were. Further changes should use the baseline
as the reference and explicitly measure the risk of losing its correct answers.

Follow-up completed on 21 September: a 72-policy CPU FTS threshold sweep over
all 1,479 labeled questions improved pooled candidate coverage from 0.972391
to 0.975434 (+0.003043). This failed the requested +0.008 prerequisite, so the
conditional append-only admission experiment was not activated. The numerical
pooled score gate at +/-0.0052 is implemented but has not passed. See
[coverage measurements and reproduction](FTS-COVERAGE-GATE.md).

Further work, not validated improvements:

1. Use saved labeled raw scores to audit candidate admission and final-ranking
   losses. Inspect a fixed sample of exclusions, amendments and near-duplicate
   statutes. Do not hand-label private changes from leaderboard feedback.
2. On CPU, measure coverage when admitting four, six or eight extra candidates.
   Score additional duo pairs only if the measured opportunity justifies it.
   More candidates can also introduce false positives; coverage alone is not
   a deployment criterion.
3. Register one narrowly defined baseline-preserving policy before evaluation.
   Report per-question wins/losses, paired uncertainty and answer-change volume.
   Account for historical development/validation reuse; a newly drawn subset
   of previously inspected data is not a genuinely untouched holdout.
4. For gold absent from retrieval, evaluate bounded complementary lexical/dense
   candidates or broader full-text matching before considering embedding
   fine-tuning and a new index. Do not fit directly on held-out misses.
5. Do not spend another full private GPU run solely because a small development
   mean increases. Agree on evidence and compute limits first.

## Reproduction references

The 22 September [supporting-authority CPU experiment](LOCAL-AUTHORITY-CANDIDATES.md)
adds name-based and yearless citation parsing. On the reused 429 development
questions, following references in the first four answer documents proposes at
most four candidates and recovers two missing gold instances. Pool coverage
rises from 0.964258 to 0.967754 (+0.003497); original raw-source text and stored
chunks agree on these coverage results. Even unlimited recognized references
reach only +0.005828, below the +0.008 prerequisite. No GPU run, private scoring
or new submission followed. The 0.9523 agreement ZIP remains the best reference.

- `reports/private/submission-review/leaderboard-outcomes.json`: reported scores
  and exact submission hashes.
- `reports/private/submission-review/development-churn.json`: replayed win/loss
  and neutral-change counts.
- `reports/private/submission-review/uploaded-zip-review-48f5bd2a6d36.json` and
  `uploaded-zip-review-352e4ae7bad4.json`: private format, provenance and differences.
- `scripts/analysis/review_development_churn.py`: validates saved score digests
  and replays development comparisons.
- `scripts/analysis/review_uploaded_private_zip.py`: read-only ZIP review. Its
  default Downloads path is machine-specific; use `--candidate` and `--other`.

Reports and most analysis scripts are ignored locally; see the artifact notes
before assuming a teammate has them in a Git clone.
