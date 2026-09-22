# Duo full-text training evidence rebuild

21 September 2026. This is **training-data preparation**, not a newly trained
model or a measured accuracy improvement. The frozen 0.9510 private submission,
checkpoint, inference modules and published GPU releases are unchanged.

**Completed and independently verified.** All 1,558 retained groups passed
archive identity, split, label, pair-preservation and evidence-budget checks.
External development/validation ID and normalized-text overlap is zero. The
fixed 16-question decoded evidence sample is saved. Five targeted tests pass.

## Findings in the original training archive

Restored the exact archive from
`gdrive:DSC-2026-results/runs/duo/duo-20260918T110105Z/artifacts/training.json.gz`.
Its SHA256 matches `training.ready.json`:

`aa09b9a985dbb6ffce06efbc19d26b2c2d7db569fecce1690a90ecda1d03c52e`.

- 1,998 question groups: 1,569 fitting questions and 429 external development
  questions in the old format. The latter were marked `development=true` and
  used for checkpoint evaluation, not gradient fitting.
- 12,824 fitting positive/negative pairs, or 25,648 oriented A/B examples.
- Fitting questions had 6 negatives (120 questions), 7 (497), or 8 (952).
  There were 113 fitting questions with multiple labeled positive documents.
- All supervised pairs compare a labeled positive against a labeled negative.
  No pair teaches both-relevant or neither-relevant behavior. Such labels must
  not be fabricated by this preparation-only change.
- The old `duo.Evidence` selects up to three article chunks and allows the pair
  tokenizer to crop them. Current `duo_evidence.Evidence` selects windows from
  the full corpus, focuses them on the query and packs each document in advance.
  The rebuilt data uses that exact current evidence implementation and the same
  verified tokenizer and 2,048-token pair limit.

The archive also contains **12 empty question/document evidence examples**:
11 among eligible training questions and one on external development. Three of
the eligible examples are gold documents. Their IDs are among `288457`,
`131890`, `263763`, and `55497`. These documents have no label or passage window
in the frozen evidence source, so no correct clause can be recovered by merely
changing passage selection. Newline-only token sequences are not usable text.
The corpus inventory has 8,512 IDs with nonempty body windows, leaving 20 known
IDs without usable body text. The corpus itself was not changed.

The rebuild quarantines all 11 eligible questions whose original pairs refer
to a document without usable corpus body. It preserves every original pair
for the remaining questions rather than silently deleting pairs or treating
empty gold documents as negatives. Exact exclusions and gold-document status
are recorded in the new input manifest.

## Evidence quality observations

Four questions from the fixed review sample were inspected while the build ran:
`134262`, `130206`, `121116`, and `134242`. They were the first available members
of the already-declared 16-ID hash sample, not selected for apparent gains.

- **130206 / 95942:** the old visible Article 28 lists all four forms of outside
  investment asked about. Current full-text evidence spends its first window
  on approval powers, then reaches Article 28 in a later window but stops after
  the first listed form. This is an observed evidence-allocation regression.
- **134262 / 46918:** both views expose Article 191, but the full-text view uses
  some budget on the preceding provision and other procedures. Different tokens
  alone are not evidence of improvement.
- **121116 / 23402:** both views emphasize pension-related cross-references,
  preservation and timing instead of consistently presenting the principal
  eligibility clauses. The document label is only the generic issuer name.
- **134242 / 269210:** the question asks about the local council's selection
  procedure; the views spend substantial space on ministerial procedure or
  eligibility/organization. Topic overlap can crowd out the requested procedure.

These examples show that training/inference evidence alignment is testable but
does **not** fix all passage-selection errors. The rebuilding task intentionally
preserves the scored full-text inference policy so the experiment remains
controlled. No claim is made that all new gold evidence is more informative,
or that the sampled document labels are wrong.

## Prepared datasets

Output directory: `reports/duo-fulltext-training/20260921-v3/`.

Completed builder identity:
`c1d9515e8d27dc3b8ccefabd1dba2ab9d5065088089d23817d35386f2e4170a5`.

| Archive | Fitting pairs, both A/B orders | Monitoring pairs, both orders |
| --- | ---: | ---: |
| Full-text control | 23,384 | 2,070 |
| Full-text plus hard negatives | 29,600 | 2,622 |

The build changed all 13,424 original question/document token sequences,
including 1,692 positive-document instances. This measures representation
changes, not improved relevance. The augmented dataset adds 3,116 negative
document instances and 3,384 positive/negative pairs before orientation.
No retained rebuilt document lacks body text. Maximum actual pair length is
2,047 tokens; the original examples reached 2,048. Across original document
instances, visible token totals changed from 12,860,247 to 11,887,246; the new
representation is not simply longer.

Full-text archive SHA256 values:

```text
control         724f396bf46e206f9d335a806d1770da78cd25f3be1cc3d9d7e46179d2d1ef0e
hard-negatives  8e77da3404d4cdc8b42f5839466367236cb3b53297678e874f059bdd858d7632
old-evidence    47a11dfc619caffbd4a4ef4a5c80fffd5d9efc877bfe94edd3e2d7eff2fe2801
```

After external exclusions and empty-source quarantine, **1,558 question groups**
remain. They are partitioned into **1,427 fitting** and **131 monitoring** groups.
All 429 external development questions and all 1,050 validation IDs and normalized
text aliases are excluded from both partitions. Duplicate training text is
deduplicated; conflicting duplicate labels would be quarantined.

The internal monitor uses SHA256 of normalized question text with the fixed
salt `fulltext-monitor-v1`, assigning remainder zero modulo ten to monitoring.
The old comparator and V9 components may already have seen these questions.
It is a monitor withheld from the new gradient-fitting partition, **not an
untouched model-level holdout**. The external development and validation sets
remain available for later ranking evaluation.

Two variants use the same partition:

1. **`control.json.gz`**: same positives, negatives, pair order and pair labels
   as the retained original examples. Only their evidence changes to the
   current full-text representation. This isolates the evidence-format change
   when compared with an old-evidence control trained on the same retained IDs
   and fit/monitor partition.
2. **`hard-negatives.json.gz`**: preserves those original pairs and adds at most
   two negatives per question. Eligible full-text phrase additions come first;
   remaining slots use high-ranked V9 top-16 candidates not already represented.
   All question gold documents and empty-source documents are excluded from
   negative selection. Each extra negative pairs with every labeled positive.

The independent verifier also exports `old-evidence-control.json.gz`, retaining
old tokenized evidence for these same usable IDs and the same new fit/monitor
roles. Its hash is recorded in `verified.json`. This makes the old-vs-new
evidence comparison possible without changing question membership or reusing
the old run's external-development checkpoint selection.

Both rebuilt variants retain the original per-question weighting convention:
each oriented pair gets `1 / (2 * number_of_pairs_for_the_question)` through
`duo.pair_examples`. Extra negatives increase examples and dilute each original
pair's weight within its question; the question's total weight stays one. That
is another reason to compare the evidence-only control before attributing a
training gain to harder negatives.

This is mining from the frozen current V9 scoring cache and current full-text
retrieval. It does **not** claim newly computed duo error scores on training
questions: those neural scores are unavailable. V9 training scores may be
in-sample. Unlabeled relevance can still make an apparent negative useful to
a real query; the dataset's gold annotations are the supervision source.

The current six-word / 0.55 full-text phrase policy is retained. The failed FTS
threshold sweep is not enabled. Evidence selection never reads the gold flag;
gold is used only to construct positive/negative training examples.

## Artifacts and verification

- `inputs.json`: frozen corpus/index/tokenizer hashes, source training identity,
  selected IDs, fit/monitor roles, exclusions, mining policy and code hashes.
- `groups/<id>.json`: resumable per-question control/augmented groups, negative
  provenance, old/new visible-token hashes, lengths, pair-budget maxima and
  the actual full-text window/token traces.
- `audit.json`: completed dataset counts, evidence differences, token statistics
  and pair counts; this file is written only after the build finishes.
- `ready.json`: completion marker and both compressed archive hashes.
- `verified.json`: independent archive/split/label/pair/token-budget verification.
- `evidence-review.json`: 16 question IDs selected by a fixed salted hash,
  showing decoded old and new evidence for all their positives and one original
  negative. Selection is independent of labels, scores and visible improvements.

Earlier `20260921-v1` and `v2` directories are incomplete diagnostic builds.
Do not train from them. The v3 builder imports only compatible completed v2
question shards with actual body text and records each imported shard hash.
The v2 process was stopped before restarting; there are no concurrent writers
to a shared output directory.

Before training, require `ready.json` **and** `verified.json`, check the archive
hashes, and inspect the fixed evidence sample. Changed token sequences, a longer
visible clause or additional negatives do not demonstrate improved ranking.

## Reproduction

Requires the local frozen assets, restored original archive and tokenizer.
The scripts do not download models or train a GPU checkpoint.

```powershell
.\.venv\Scripts\python.exe -m unittest tests2.test_fulltext_training
.\.venv\Scripts\python.exe scripts/analysis/rebuild_duo_fulltext_training.py --workers 8 --reuse-prepared reports/duo-fulltext-training/20260921-v2
.\.venv\Scripts\python.exe scripts/analysis/verify_duo_fulltext_training.py
```

To build from scratch elsewhere, omit `--reuse-prepared` and use a new `--out`
directory. Both builder and verifier accept `--out`. A resumed build must keep
the same inputs and import configuration. The evidence and model code in the
frozen modules is not modified by this task.

`development` in the rebuilt group schema means **internal monitoring**, to
match `duo.pair_examples`; the explicit `role` field removes ambiguity. These
archives must not be copied over the old run's `training.json.gz`. A future
trainer needs a new manifest/run ID, correct fit/monitor identities, and model
selection that does not reuse old epoch/checkpoint state. Compare old-evidence,
full-text control and augmented data using the same starting checkpoint and
training settings before attributing differences to the rebuilt examples.

No GPU training is authorized or launched by the preparation scripts. A/B is
unchanged; a neither/both objective would be a separate model and evaluation
change. Any eventual submission still needs the requested labeled score gate.

## GPU handoff and design-review corrections

The additive `legalir2.fulltext_retrain` runner and
`scripts/gpu/gpu-fulltext-retrain-4090.sh` now implement the controlled experiment.
See [the 4090 runbook](../runbooks/RUNBOOK-fulltext-retrain-4090.md) for setup,
stages, resumption and output paths. No GPU training has been performed locally.

Default arms are `old-evidence-control` and `control`: same retained pairs,
same 1,427 fit / 131 monitor questions, same initialization and training settings.
Both are evaluated with identical full-text inference tokens, priors and pools.
The evidence intervention is therefore isolated. The third archive
`hard-negatives` is optional and exploratory; comparing it with `control` would
measure added negatives. It is not selected by this run's release gate.

The 131 monitor questions select an epoch by pair loss only; neither their loss
nor their recall is evidence of a private-score gain. Historical models may have
seen them. The 429 development screen can only stop computation or continue to
the 1,050 validation questions. No split summary contains a `decision`.

`labeled_gate` now reports deterministic paired bootstrap intervals and exact
sign-test probabilities. Only the complete 429+1050 pooled report can decide:
delta must cross the requested +/-0.0052 deadband and its 95% paired interval must
exclude zero. The GPU release additionally requires a positive paired interval
against the matched old-evidence arm before attributing a baseline win to the
evidence intervention. Historically reused evaluation labels remain a limitation.

The support filter now raises immediately if the training evidence object lacks
its explicit `supported` document set. There is no whole-corpus fallback. This
guard does not change the already verified v3 archive bytes or frozen inference.
