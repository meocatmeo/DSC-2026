# v11 experiment: richer document evidence toward 0.96

Written 16 September 2026. Status: **initiated**. Baseline artifacts recovered using
the user-specified `aic-readwrite.conf`; offline audit and deterministic evidence
builder implemented. No new training, GPU inference or submission has been run.

## Execution checkpoint

Resume update: per-question atomic checkpoints, input/configuration guards,
automatic reuse of local RUN_ID, remote checkpoint restore and periodic checkpoint
upload. Optional dense restore falls back to rebuild; missing required splits or
trained g32 fail with explicit errors when no local copy exists.

17 September update: user requested GPU setup without blocking on historical corpus
verification. `gpu-v11.sh` and `legalir2/v11.py` now provide benchmark, collection,
matched exploratory fusion and paired evaluation. See `RUNBOOK-v11.md`. Active
corpus metadata is logged; historical fingerprint mismatch does not block execution.
The original untouched-calibration protocol remains unverified; this ready-to-run
variant is explicitly exploratory. No GPU measurement has yet been executed here.

- Corpus recovery is frozen at 19 experimental text replacements; 30 of the 49
  truncated contexts are deferred. See `analysis/corpus-repair/refined49/REPORT.md`.
- v9 predictions, diagnostics, manifest, metrics and stacker recovered into
  `analysis/v11/baseline/`. Recomputed development Recall@5: **0.9463492063**.
- Current Drive `corpus.jsonl.gz` does not match v9's recorded fingerprint;
  608 passages also differ from local after whitespace normalization. Do not
  assume this remote file is the exact baseline source; resolve or reproduce v9.
- 85 missed gold documents: 36 outside the 50-candidate reranked pool, 49 inside.
  The latter remain unresolved until actual model inputs are replayed.
- `analysis/v11/audit_baseline.py` checks prediction/diagnostic/split agreement,
  corpus identity and per-error recall mass. See `analysis/v11/BASELINE-AUDIT.md`.
- `legalir2/evidence.py` builds deterministic six-passage/2,048-token bundles;
  `Reranker.score_bundles` scores their exact IDs without another truncation pass.
  These are opt-in components; the deployed prediction path remains unchanged.
- Tests: 109 completed, 84 passed and 25 skipped because PyTorch is unavailable in
  the verification runtime. Real-tokenizer/GPU benchmarking remains pending.
- Next: resolve baseline corpus/checkpoint provenance, replay actual inputs,
  verify calibration membership, then benchmark on training questions. Keep the
  corpus-only comparison separate from the richer-evidence comparison.

**New prerequisite from the corpus audit:** `analysis/CORPUS-COMPLETENESS.md` confirms 20 empty source passages and flags 49 potential truncated previews. The local index preserves the same source fingerprint and has zero body text for all 20 empty documents. Obtain and evaluate corpus repairs before paying for richer-evidence inference; longer inputs cannot recover absent source text. The quantified error overlap currently uses v4 diagnostics, not v9.

## Objective and decision

Try to improve on v9's reported public Recall@5 of 0.9508 by testing whether the Qwen partner is missing decisive evidence under the current passage representation. The target is 0.96, requiring +0.0092 absolute recall, or about 18.7% of the remaining public recall loss. This is a target, not an expected result.

The next experiment is a current-v9 error audit followed by one controlled richer-evidence comparison. Keep v9 as the deployment baseline until the complete alternative has earned promotion.

## Evidence and limits

- SUMMARY.md identifies v9 as the two-reranker, 20-feature stacker configuration; reported validation recall is approximately 0.9464.
- ANALYSIS-v7.md measured v4 candidate recall@50 of 0.980476 against final recall of 0.945397. Its 86 missing gold documents and 50 misses inside the candidate pool describe v4, not v9. Recompute those numbers before using them to prioritize work.
- Candidate recall is an oracle bound. It does not show that the passages actually sent to a model contain enough evidence to identify the gold document.
- Document-level loss has already produced negative results from both g32 and base initialization, according to PROCESS-v6.md. Repeating those experiments is not this plan.
- The Qwen3-Reranker-0.6B model card documents a 32K context window: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/main/README.md. This establishes input capacity, not a benefit from longer inputs or acceptable throughput on the available hardware.

## 1. Establish the actual v9 baseline

Locate the saved v9 prediction, diagnostics, run manifest, stacker, model provenance, and corpus fingerprint. Likely artifact names include outputs/v9-ensemble.diagnostics.json and outputs/v9-stacker-ens.json; verify their presence and contents rather than inferring configuration from script defaults.

Record model/checkpoint identities, parameter counts, retrieval settings, rerank depth for both models, passage selection, token limits, prompt format, stacker feature order, question split, and corpus identity. Recompute macro Recall@5 from the predictions. If any artifact is missing, recover it or reproduce the baseline before comparing variants.

Save an audit with one row per missed gold document:

| Field | Purpose |
|---|---|
| Question ID and number of gold documents | Preserve the macro-recall contribution, 1 / number of golds |
| Gold ID and original candidate rank | Separate retrieval misses from ranking misses |
| Final rank and fifth-place competitor | Identify the selection error |
| Passage IDs and actual tokenized input | Inspect what each reranker could see |
| Truncated or omitted text | Identify possible evidence loss |
| Error category and supporting excerpt | Make the diagnosis reviewable |

Use four categories: outside the reranked pool; relevant evidence omitted or truncated; relevant evidence visible but ranked incorrectly; ambiguous or unresolved. Do not force uncertain examples into a category. A human reading a gold document may discover useful passages, but gold-guided selection must never become the evaluated inference procedure.

Report error mass by category, not only counts. An unseen positive document is not automatically unlearnable: text-based retrieval can generalize to documents absent from training positives.

## 2. Prespecify the richer-evidence variant

Freeze retrieval, candidate IDs and ordering, the first reranker, Qwen weights, query instruction, and top-five output rule. Score the same candidate depth as verified for v9. Change only Qwen's document representation before the subsequent fusion comparison.

Construct one deterministic document bundle per question/candidate:

1. Include the corpus-provided document title and available article headings. Do not use auxiliary annotations absent from the deployment corpus.
2. Select up to six distinct query-relevant passages using the existing retrieval evidence, without gold labels. Define and freeze tie-breaking and overlap suppression.
3. Pack passages in retrieval-score order, with headings and clear separators. Use a 2,048-token total input budget, including the question, instruction, and model-specific suffix. Reserve required prompt tokens before allocating document space.
4. Record selected chunk IDs, dropped chunks, token counts, and truncation. Preserve the model's required scoring template.

Start with 2,048 tokens as the single primary variant. A 4,096-token run is a later experiment only if the audit shows decisive evidence still being truncated and measured cost is acceptable. Do not sweep passage counts, context lengths, prompts, and fusion choices together.

Benchmark throughput and memory on a small training-only sample before a full scoring pass. Estimate total runtime from measured examples and candidate counts; do not promise GPU hours in advance.

## 3. Handle fusion without repeating the v10 confound

One bundled score is not interchangeable with v9's multiple passage scores. Do not silently feed it into the existing max/second/mean/spread feature block or assume the old weights remain calibrated.

Cache the original v9 features and the new Qwen bundle score. Define a versioned feature schema that retains the old features and adds the bundle score as an explicit feature. This tests whether the new evidence adds information while preserving the existing signals; report the extra scoring cost.

Fit two matched fusion models using the same fitting method, regularization, labels, and splits: a control with the original features, and a treatment with the additional bundle feature. Compare both with frozen deployed v9. The control distinguishes effects of refitting from effects of the new evidence.

Fusion fitting requires questions excluded from training of the models producing their scores. Verify g32 training membership; do not assume a stacker-only split removes reranker leakage. If an adequate untouched calibration set does not exist, budget for properly grouped cross-fitting or explicitly label a cheaper comparison as exploratory. Do not transfer calibration weights from a half-data weaker reranker and treat that as a clean test on the full-strength reranker.

Group duplicate normalized question text together. Evaluation labels must be excluded from fusion fitting and every neural training stage. The existing 1,050-question split has already informed development; call its result a development comparison, not an untouched test.

## 4. Evaluate all questions and decide

Run the complete control and treatment on all development questions, including those v9 already answers correctly. Save predictions, per-question recall, model inputs, settings, and timings.

Report:

- Macro Recall@5 and paired deltas against both frozen v9 and the matched refit control.
- Rescues, regressions, and their recall-weighted contributions.
- Paired bootstrap interval for the mean delta; do not apply a universal detection floor.
- Single-gold and multi-gold results; retrieval, evidence-loss, and visible-evidence buckets.
- Latency, peak memory, parameter count, and artifact provenance.

Use these decisions:

| Result | Action |
|---|---|
| Treatment improves on the matched control and v9, with credible paired evidence | Confirm on a stronger held-out or outer grouped evaluation before a score claim; prepare a candidate submission if the result and cost justify it |
| Positive mean but uncertain comparison | Keep as exploratory; obtain more independent evidence instead of selecting a lucky sweep winner |
| Evidence-loss cases improve but regressions erase the benefit | Inspect the regression mechanism before proposing another variant |
| Decisive evidence is visible and richer input does not help | Shift the next hypothesis to reranker discrimination/adaptation |
| Material error mass lies outside the candidate pool | Design a retrieval experiment targeting those misses and measure conversion to final recall |
| Treatment fails to add value | Retain v9 and close this experiment |

A gain near +0.0092 on development would be encouraging, but it does not predict public 0.96. Do not add a fixed historical validation-to-public offset. Do not train on all labels until model selection is complete, then reuse those labels as a holdout.

## Deliverables and implementation order

1. Verified v9 manifest and reproducible current error audit.
2. Shared, deterministic bundle builder with token and passage provenance.
3. Training-only throughput measurement and full-run cost estimate.
4. Original-feature and bundle-score caches, versioned fusion schema, matched fitting protocol.
5. Full paired evaluation report and an explicit continue/stop decision.
6. Only after a favorable decision: validated prediction archive and deployment provenance.

Implemented interfaces: `legalir2/evidence.py` builds the bundle;
`Reranker.score_bundles` scores exact prompt IDs; `legalir2/v11.py` adds the bundle
as column 21 beside the unchanged 20-feature v9 matrix. The dedicated CLI is
`python -m legalir2.v11`, not a bundle flag on `legalir2 predict`. `gpu-v11.sh`
orchestrates it. Production `stack.py` and normal v9 prediction remain unchanged.

Membership audit (17 September): restored `train-r2.jsonl` contains all 5,950 train
question IDs, leaving **zero** untouched training questions even after normalized
text grouping. `gpu-phase2.sh:153` identifies that file as the g32 training input;
`train-doc.jsonl` belongs to the later document-loss experiment. The checkpoint
metadata agrees on group count but does not bind the file by hash, so this is
recipe-based reconstruction. See `analysis/v11/membership-audit.json`.
The available runner therefore implements an exploratory calibration comparison,
not the clean cross-fitted protocol. No clean-calibration or GPU-runtime claim is made.

This document authorizes no new score claim: the route to 0.96 remains a hypothesis until the complete pipeline demonstrates a gain.
