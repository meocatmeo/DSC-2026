Layout-v2: scripts, data and documentation now live in folders; cloud runtime paths are under assets/ and runs/. See ../PROJECT-LAYOUT.md.

# Active submission target: private test

Developer handoff: [overview](../development/README.md),
[experiment results](../development/EXPERIMENT-RESULTS.md), and
[artifact/operations notes](../development/ARTIFACTS-AND-OPERATIONS.md).

The user switched all future test/submission work to the private test on 19 September 2026.

- Workspace input: `data/private-official.json`.
- Original: `C:/Users/LENOVO/Downloads/private-official.json`.
- Questions: **2,080**, each with a nonempty question and `answer: null`.
- JSON keys verified unique. No question IDs overlap `public-official.json`.
- SHA256: `9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc`.
- Original and workspace copy have identical hashes.
- Machine-readable record: `analysis/private/input-manifest.json`.

## Current best: full-text duo, 20 September 2026

Latest outcome (2026-09-21): the deep-fusion private submissions regressed.
The user reported "216 only got 0.9464 and 512 got 0.9459"; "216" is interpreted
as the previously identified 256-question submission, pending correction.
The 256 ZIP SHA256 is `48f5bd2a6d363800a083874e52b3bf572c8ba91b8080f82205124bbc8ae57d15`;
the 512 ZIP SHA256 is `352e4ae7bad4d5f469fa2f22c77d8500d17f04443dfc06316256da34b028c877`.
Their private scores are respectively 0.0046 and 0.0051 below 0.9510. These are
user-reported leaderboard outcomes, not locally recomputed private metrics.
Do not promote either deep-fusion candidate based on its development score.
Both archives passed schema, corpus-ID, question-coverage and run-hash checks.
Development improvements did not generalize; private error identities remain
unknown. See `reports/private/submission-review/leaderboard-outcomes.json` and
`development-churn.json`. Preserve the scored full-text baseline and all caches;
no further GPU run has been authorized by this score report alone.

The user reported **0.9510** for the full-text duo from
`private-duo-fulltext-v2-4090-fast1`, an improvement of **0.0041** over the
original duo. Preserve the exact ZIP at
`submissions/private/20260920-fulltext-v2-09510/duo/submission-private-duo.zip`.
Its SHA256 is `40caa611cfdbf4f853296fdfef93774b117fffc4c339d591690080a59e0a6d57`.
The new V9 predictions are identical to the original V9 submission.

Audit: `reports/private/20260920-scored-audit/REPORT.md`. All 2,080 answer sets
pass format/membership checks. Mechanical checks of 80 cached questions and
1,389 document traces found no inconsistent ranking or invalid evidence budget.
Eight selected questions received diagnostic comparison; there are no private
gold labels. The local cache-consistency fix leaves the scored ZIP and released
ranking policy unchanged. Do not replace files in a running/signed workspace.

Further bottleneck analysis: `reports/private/20260920-score-bottlenecks/REPORT.md`.
All 2,080 private rankings were replayed. The full-text expansion added 2,929
candidate occurrences but none reached the submitted top five; 174 were in
duo's ability top five. Restoring 1,223 available discarded V9 scores changes
21 sets in an unscored CPU experiment and selects only two added candidates.
The next substantive experiment is a full-text pointwise prior for every
candidate, calibrated with duo on labeled data. Do not treat diagnostic
counterfactuals or the unscored replay as demonstrated score improvements.

The separate 4090 experiment now combines full-text relevance with calibrated
tournament consensus from teammate branch `thong/e091868`. See
`docs/state/THONG-INTEGRATION.md` for the branch audit and exclusions, and
`docs/runbooks/RUNBOOK-fulltext-prior-4090.md` for its launcher. No new private
score has been measured; 0.9510 remains the best reported result.

The full-text-prior GPU run retained baseline on development (0.9502719503).
Follow-up CPU retrieval work and its measured limits are documented in
`docs/state/RETRIEVAL-AFTER-PRIOR.md`. Keep BF16 precision; the user explicitly
requested no quantization. The new retrieval experiments are not deployed.

## Original private baseline

The user supplied both private prediction JSONs and reported leaderboard scores:
**duo 0.9469**, **V9 0.9457**; duo leads by **0.0012** (0.12 percentage points).
Both files cover exactly the 2,080 official IDs with five distinct answers each.
Their answer sets differ for 1,001 questions, replacing 1,124 document slots; another
594 questions differ only in order. Private gold labels are unavailable, so these
differences cannot be labelled as individual improvements or regressions.
File hashes and detailed changes: `analysis/private/leaderboard-comparison.json`.
Use the 0.9510 full-text duo above as the current best private baseline. Existing public scores
and manual submissions are historical and must not be presented as private results.
Do not carry public-ID answer edits into this test.

Organized deliverables: `submissions/private/20260919-v1/{duo,v9}/` locally and
under `gdrive:DSC-2026-results`. GPU package: `releases/private/20260919-layout-v2/`.
See `docs/PROJECT-LAYOUT.md` for the complete directory map.

Use `outputs/private/` for private submissions and run-scoped `cache/v9` and
`cache/duo` beneath the private output directory. Corpus indexes and frozen model weights may be reused
when their fingerprints match; query scores and resume state must match this input.
Do not overwrite the historical public file to make an old runner consume private data.

**Private inference is prepared** in `legalir2/private_infer.py` and
`scripts/gpu/gpu-private-h100.sh`; see `docs/runbooks/RUNBOOK-private.md`. It produces separate V9 and duo ZIPs,
reuses V9 private scores for duo, and restores saved assets through rclone. The user
requested both checkpoints. No actual private GPU inference has run locally.
The older duo CLI remains public-specific; do not replace its historical question
file or change its signed trained modules. The new entry point verifies model/index
identity while giving the private file its own resume signature.

The prior evidence audit is complete at
`analysis/v11/investigation/miss-audit/REPORT.md`: 49 missed gold-document instances,
11 with relevant omitted text, 28 with substantive evidence already visible, and 10
uncertain. No main-g32 passage token truncation was found in those cases. These are
diagnostic validation findings, not private-test performance measurements.
