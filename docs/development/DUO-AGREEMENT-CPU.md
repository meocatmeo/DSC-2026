# CPU experiment: preserve four baseline answers, require agreement for the fifth

22 September 2026. An experimental ranking rule is implemented in
`legalir2/duo_agreement.py`. It reuses the original full-text comparator and
retrained control's cached pair probabilities. No GPU inference, validation
run, private scoring, submission creation, or production policy change was
performed. Keep the 0.9510 private baseline.

## Rule

1. Reconstruct the baseline using its unchanged 0.25 blending strength.
2. Preserve its first four documents, including their order.
3. Consider outsiders already in the shared candidate pool. An outsider is
   eligible only if **both** comparators assign it at least 0.90 pair preference
   over the baseline's fifth document, and both give it higher fitted ability.
4. Make at most one replacement. Choose the eligible outsider with greatest
   minimum pair preference across models, then minimum ability advantage, then
   document ID. Otherwise keep all five baseline answers.

The 0.90 value is a fixed model-output threshold, not a calibrated 90% probability
of legal correctness. The two models share lineage and evidence; agreement is
not independent confirmation. All outsider evidence was already GPU-scored.
The predictor does not read gold labels or contain query/document-specific rules.

## Measured result on existing development

| Method | Macro recall | Changed sets vs baseline | Replaced slots |
|---|---:|---:|---:|
| Full-text baseline | 0.9502719503 | 0 | 0 |
| Retrained control | 0.9526029526 | 130 | 139 |
| Agreement fifth-slot experiment | 0.9526029526 | 71 | 71 |

Against baseline, the experiment has **one win, zero losses**, +0.002331 recall,
paired bootstrap 95% interval **[0, 0.006993]**, sign-test p=1.0. Seventy of the
71 changed sets are recall-neutral. It reduces set changes by 45.4% versus full
replacement with the retrained model while matching its aggregate development
recall. Fewer changes do not establish lower private error.

The gained question differs: **88726**, the tax-installment procedure, replaces
139874 with gold 160381. Both pair preferences exceed 0.999 and the minimum
ability advantage is 2.687. The model's low prior previously kept that gold at
rank 13. The retrained model's sole baseline gain, **162906**, is not retained by
this rule. Thus against the retrained control the rule trades one win for one
loss; it has not demonstrated a higher overall recall than that model.

This policy was motivated by errors already inspected on these 429 questions.
The fixed rule and code hash were recorded before running its evaluation, and
no threshold sweep was performed, but the evaluation is **not independent**.
The one-question result remains inconclusive and does not authorize promotion
or predict a private score increase. Do not keep adjusting thresholds to fit
these two named cases.

## Candidate admission follow-up

The earlier audit recorded positions within the cached V9 candidate lists.
Replaying V9's weighted scores gives these actual ranking positions:

| Development question | Gold document | Cached list position | Weighted V9 rank |
|---|---|---:|---:|
| 162336 | 225130 | 46 | 42 |
| 52892 | 237914 | 18 | 20 |
| 108738 | 11864 | 26 | 40 |
| 60178 | 237914 | 29 | 27 |
| 71570 | 74416 | 31 | 27 |

All are below the comparator's top-16 cutoff. None passes the existing memory
rescue: four have no supporting memory match; 60178's document is fifth in
memory ranking with similarity 0.3344, below the top-three/0.6 admission rule.
Recorded phrase additions also omit these gold documents. This is the current
selection policy operating as coded, not an off-by-one or missing-rank bug.

Increasing depth to 20 would expose only one of these five documents; reaching
all five by depth alone requires at least 42. That would require many new pair
scores and reproduce the risk of the failed broad deep-pool experiment. A more
selective complementary candidate channel is the appropriate next research
target; this CPU rule cannot score documents outside the saved pool.

## Reproduction and checks

```powershell
.venv\Scripts\python.exe -m unittest tests2.test_duo_agreement -v
.venv\Scripts\python.exe scripts/analysis/evaluate_duo_agreement.py
```

Four tests cover protected answers, agreement/ability rejection, malformed input
and label independence, and candidate-order invariance. The real evaluation
checks source-row digests and fitted scores, shared priors/pools, 429-question
membership count and the exact saved development score digest. Reports under
`reports/duo-agreement-v1/` contain the fixed plan, all changes, paired statistics
and five admission traces. The script never reads private prediction caches or
the 1,050 validation score set.

To deploy this rule later on newly computed data, both comparator scores must
exist. Cached CPU replay has no added model execution cost; fresh inference
would require both comparators and a new parameter/runtime audit. No GPU release
was published for this experiment.

## User-requested private trial, 22 September

After the development diagnostic, the user asked to try the unchanged rule.
`scripts/analysis/submit_duo_agreement.py` applies that exact recorded code hash
and policy to both saved private caches. No threshold adjustment, GPU inference
or validation run was performed. This is a later private prediction-generation
step; it does not change the original development plan's scope.

The resulting ZIP is
`submissions/private/20260922-duo-agreement-v1/submission-private-agreement.zip`.
All 2,080 questions contain five distinct known corpus IDs. Exactly 306 answer
sets differ from the 0.9510 baseline, always by replacing only the fifth answer;
the first four and their order are preserved for every question. It differs on
891 sets from the 0.9500 retrained submission. The original ZIPs remain intact.

SHA256: `37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f`.
The output directory includes `inputs.json`, `review.json`, `changes.json` and
the ZIP checksum. Frozen baseline shard hashes, previously audited candidate
shard hashes, source identities, saved fitted-score gradients and exact source
ZIP replay were verified before packing. The final ZIP was reopened and its
payload compared with the predictions. The file was prepared for manual upload.

## Reported private result

The user reported **0.9523** on 22 September for this candidate: **+0.0013**
versus the previous best 0.9510, and +0.0023 versus the retrained-only 0.9500.
This is now the best reported private submission. The ZIP checksum was rechecked
after the score report and is unchanged. `leaderboard-result.json` alongside
the ZIP records the result separately from the original pre-submission review.

Preserve the exact rule, both score caches, and the original 0.9510 ranking
anchor. Do not apply the rule repeatedly to its own output. Private labels
remain unavailable: this aggregate score does not reveal which replacements
won or lost, nor establish a gain on a new test distribution. The earlier
development-only uncertainty statements describe the evidence available before
this private result; no statistical gate is retroactively marked as passed.
