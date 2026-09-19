Layout-v2: scripts, data and documentation now live in folders; cloud runtime paths are under assets/ and runs/. See ../PROJECT-LAYOUT.md.

# Active submission target: private test

The user switched all future test/submission work to the private test on 19 September 2026.

- Workspace input: `data/private-official.json`.
- Original: `C:/Users/LENOVO/Downloads/private-official.json`.
- Questions: **2,080**, each with a nonempty question and `answer: null`.
- JSON keys verified unique. No question IDs overlap `public-official.json`.
- SHA256: `9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc`.
- Original and workspace copy have identical hashes.
- Machine-readable record: `analysis/private/input-manifest.json`.

The user supplied both private prediction JSONs and reported leaderboard scores:
**duo 0.9469**, **V9 0.9457**; duo leads by **0.0012** (0.12 percentage points).
Both files cover exactly the 2,080 official IDs with five distinct answers each.
Their answer sets differ for 1,001 questions, replacing 1,124 document slots; another
594 questions differ only in order. Private gold labels are unavailable, so these
differences cannot be labelled as individual improvements or regressions.
File hashes and detailed changes: `analysis/private/leaderboard-comparison.json`.
Use duo as the current best reported private baseline. Existing public scores
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
