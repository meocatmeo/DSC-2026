# Comparator result: 0.9508 → 0.9500

The user-reported public score fell by 0.0008 (0.08 percentage points). Keep the original V9 submission as the best known result. This analysis uses saved scores, predictions and logs only; it runs no neural inference. Public ground-truth labels are unavailable, so it cannot identify which public replacements were wrong or attribute the public loss to one component.

## What actually changed

| Comparison | Changed answer sets | Recall wins | Recall losses |
| --- | ---: | ---: | ---: |
| Development comparator vs replay, 429 questions | 220 | 4 | 2 |
| Validation comparator vs replay, 1,050 questions | 503 | 5 | 3 |
| Public comparator vs replay, 1,000 questions | 451 | Unknown | Unknown |
| Public submitted comparator vs original saved V9 | 454 | Unknown | Unknown |

The validation report's “1,042 identical” means equal per-question recall, **not identical predictions**. Of 503 changed answer sets, 495 had unchanged recall. Many replacements changed non-gold documents while retaining the same gold documents. The algorithm therefore made substantially more interventions than the original report implied.

## The baseline was not an exact replay

Before adding the comparator, the regenerated pointwise baseline already differed from the saved V9 submission on **91 public questions**. Of these, 26 received no further set change from the comparator, 65 received changes from both components, and 23 ended up back at the original V9 set. Another 386 questions changed only through the comparator. These counts reconcile to 454 final differences from original V9.

The same issue occurs on validation: 84 baseline answer sets changed, with zero recall wins and one partial-credit loss. Original V9 validation recall was 0.946349; the replay scored 0.946032. The comparator's 0.948254 is therefore +0.002222 over the replay, but only **+0.001905 over the original saved V9**.

The original runner allowed this recall drift within its tolerance. Matching model/index hashes and settings did not establish identical predictions. The cached outputs prove the mismatch, but do not establish whether its cause was numerical behavior, batching, or a scoring implementation difference. Do not describe the public experiment as an isolated comparator comparison until this is resolved.

## Where validation gained and lost

| Question ID | Change in per-question recall | Added document | Removed document |
| --- | ---: | --- | --- |
| 7390 | +1 | 163084 | 162150 |
| 159014 | +1/3 | 195962 | 227748 |
| 112448 | +1 | 128021 | 79407 |
| 14554 | +1 | 61357 | 83821 |
| 102328 | +1 | 42262 | 195377 |
| 64354 | −1 | 70582 | 234757 |
| 152976 | −1/2 | 244431 | 77650 |
| 41690 | −1/2 | 50885 | 242728 |

The three losing swaps were assigned probabilities of 0.9661, 0.9966 and 0.9860 in favor of the replacement. Those probabilities do not establish confidence calibrated to the dataset's labels. A 95% probability threshold would retain all three mistakes; even a 99% threshold retains one.

For question 41690, about penalties for violating admission eligibility rules, the gold-labeled document removed was `242728`, whose title identifies an education administrative-penalties amendment. The added document `50885` is titled as road/rail traffic administrative penalties. For question 152976, the replacement was an adoption-department organization decision, while the removed gold document concerned foreigners' entry/residence. These are observations about dataset labels and document titles, not a legal interpretation of the underlying provisions. They warrant inspecting the actual selected passages before another training run.

## More influence and more epochs did not help

| Comparator blend strength | Development recall | Validation recall |
| --- | ---: | ---: |
| 0 | 0.949689 | 0.946032 |
| 0.25, selected on development | 0.951437 | 0.948254 |
| 0.5 | 0.947164 | 0.945714 |
| 1 | 0.909868 | 0.923810 |
| 2 | 0.839161 | 0.861429 |

Validation values at the other fixed strengths were calculated retrospectively for diagnosis, not for choosing a replacement submission. Strength 1 uses the regularized tournament abilities, which still contain the baseline prior; it is not a pure comparator with no baseline contribution.

Held-out pair loss worsened from **0.46328 at epoch 1 to 0.66467 at epoch 2**, despite very low late training losses. This is consistent with overfitting. The training code selected the adapter with the lower development loss, so it should already have merged epoch 1. Simply switching to epoch 1 is not a new remedy.

The training examples compare gold documents against negatives. Inference compares all candidate pairs, including negative-versus-negative pairs not represented by that training objective. Coupled with a forced A/B choice, this is a plausible source of excessive rearrangement; it is a hypothesis, not a demonstrated cause of the public regression.

## Next work, before renting another GPU

1. Establish an exact original-V9 control using its saved public answers and validation diagnostics. Audit the 91 public replay differences before using regenerated scores as the reference for another submission. Keep original predictions unchanged when an intervention is not explicitly applied.
2. Inspect selected passages for the eight recall-changing validation cases. Check whether the labeled support was absent/truncated or the comparator chose poorly despite seeing it. These require different fixes.
3. If retraining is justified, target the top-five inclusion decision: positives missed near the cutoff, multiple gold documents, and the actual confusing alternatives. Consider an explicit neither/both relevance treatment rather than forcing all irrelevant pairs into a winner. Evaluate the choice on a genuinely held-out split before paying for public inference.
4. Use development data to test a limited-swap policy as an ablation against broad tournament reranking. Do not tune a confidence threshold to remove these three known validation losses or use the public score as a label for individual questions. High raw A/B probability alone is not supported as a reliable gate by this run.

No new submission was generated or uploaded during this postmortem, and there is no demonstrated score gain from the proposed next steps.

## Reproducible artifacts

- `analysis/impact/analyze_duo.py`: CPU-only analysis script; asserts question coverage and reconstructs the submitted answers from saved scores.
- `analysis/impact/duo-20260918T110105Z/postmortem.json`: all counts, 454 public changes versus original V9, the 91 baseline replay differences, and document titles.
- `analysis/impact/duo-20260918T110105Z/postmortem.md`: detailed question text and per-swap probabilities/margins for changed sets.
- `analysis/impact/duo-20260918T110105Z/training-run.log`: the saved training log.

Recompute with `.venv/Scripts/python.exe analysis/impact/analyze_duo.py` from the project root. The saved validation oracle remains 0.975714; it describes candidate coverage only and does not establish attainable trained performance.
