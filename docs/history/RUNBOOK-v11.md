# V11 GPU run

## Automatic rclone workflow

The release is staged in `_upload` and published to `gdrive:DSC-2026`.
On a GPU box with Python 3 and an authenticated `gdrive` rclone remote, run:

```bash
mkdir -p /workspace/DSC-2026
cd /workspace/DSC-2026
rclone copyto gdrive:DSC-2026/gpu-v11-start.sh gpu-v11-start.sh
bash gpu-v11-start.sh
```

Run inside tmux to survive SSH disconnects. The starter downloads `v11-release.zip`;
the autopilot restores splits, corpus, embedding and g32; downloads Qwen if missing;
sets up the existing pinned environment and builds missing/incompatible indexes.
It then runs benchmark, collection, matched fitting, evaluation and public prediction
automatically. It creates and uploads `submission-v11.zip` (containing exactly
`submission.json`) with the fitted treatment. It does not rent a box or send the ZIP
to the competition website. Validation improvement is not assumed or required to
produce this candidate archive.

Each run uses a timestamp ID. Logs/heartbeat are uploaded every 60 seconds to
`gdrive:DSC-2026-results/logs/`, like the other phase scripts; caches and results are copied after each
stage and on exit to `gdrive:DSC-2026-results/v11/<RUN_ID>/artifacts/`. A failed upload is reported and makes final
exit fail. The startup log-channel probe must pass before expensive work.

```bash
rclone lsf gdrive:DSC-2026-results/v11 --dirs-only
rclone cat gdrive:DSC-2026-results/logs/STATUS.txt
```

Optional: `RUN_ID=myrun BUNDLE_BATCH=1 LIMIT=2000 bash gpu-v11-start.sh`.
Rerunning on the same box reuses `outputs/.v11-active-run` automatically. On a new
box, set the previous `RUN_ID` to restore its checkpoint rows from Drive. Set a new
RUN_ID only for a genuinely new experiment. Completed questions are skipped;
changing model/configuration/data/code rejects mixing incompatible checkpoints.
`BUNDLE_BATCH` may be lowered on resume after OOM.
Each question is saved as an atomic gzip file under `<split>.rows/`; these immutable
rows are backed up every 300 seconds (`CHECKPOINT_EVERY`) as well as on exit.
The uploader freezes complete index lines before transferring, uploads immutable
rows with checksum verification, and publishes the frozen index only after those
transfers succeed. It never sends the actively appended index directly.
Drive operations print progress every 30 seconds and have bounded retries and a
30-minute wall-clock limit (`V11_TRANSFER_TIMEOUT`). Background checkpoint transfers
use five minutes per operation; log transfers use two. The heartbeat continues
during the final backup. Completed splits skip checkpoint upload and restore.
Starting a second updated V11 launcher in the same workspace is refused before
release extraction; this lock does not protect processes launched by older scripts.
Hard machine loss can lose work since the last successful cloud backup; ordinary
process failure retains all committed local rows. A partially written row is ignored.
Credentials are configured separately on the GPU and are never in `_upload`'s
V11 release. Local publishing uses `aic-readwrite.conf` explicitly.

## Manual stages on an existing v9 box

Use the existing v9 GPU environment (`.venv`, unchanged dependencies). Historical
corpus verification is not a prerequisite, as requested. The active store and model
identities are recorded with each cache. A stale dense/store pair still needs to be
rebuilt because its chunk IDs would be incompatible.

Required existing files: `artifacts2/store.sqlite`, `artifacts2/lexical/`,
`artifacts2/dense/`, `models2/embedding/`, `models2/reranker-g32/`,
`models2/partner_qwen/`, and `artifacts/splits/{train,validation}.json`.
The release archive contains the updated code and the downloaded v9 run/stacker
files under `analysis/v11/baseline/`. Extract it into the GPU project root.

```bash
cd /workspace/DSC-2026
source .venv/bin/activate
python -m unittest tests2.test_evidence tests2.test_v11 tests2.test_v11_resume tests2.test_v11_collect tests2.test_v11_preflight -v
bash gpu-v11.sh benchmark
bash gpu-v11.sh collect
bash gpu-v11.sh fit
bash gpu-v11.sh evaluate
bash gpu-v11.sh submit
```

Or run all stages on a fresh output directory:

```bash
OUT=outputs/v11-run1 BUNDLE_BATCH=2 LIMIT=2000 bash gpu-v11.sh all
```

The benchmark scores ten training questions and reports elapsed time, projected
validation runtime, and GPU peak allocation. Default Qwen bundle batch is 2; set
`BUNDLE_BATCH=1` if necessary. Baseline reranking uses its frozen batch/configuration.
Caches contain exact bundle token IDs and original passage strings and may be large.
After a failed/OOM run, rerun with the same `OUT`; scoring resumes from committed
question rows. The final JSONL gzip is assembled only after collection completes.
The September 17 repair explicitly allows resume from the reviewed code hashes
listed in `legalir2/v11_compat.py`; unknown scoring changes still fail closed.
Older completed caches lack a full saved resume signature. Their recorded input
metadata is checked and a warning states that original code/weight hashes cannot
be revalidated. New completed caches retain the full signature and compressed size.
`fit` and `evaluate` reuse completed caches without loading GPU models. Earlier
releases' monolithic partial gzip files cannot be resumed by this row-checkpoint
format; they are retained until the newly collected complete file replaces them.

`collect` runs 2,000 training questions (after excluding validation IDs and duplicate
normalized validation text), then all validation questions. Both original rerankers
score the same 50 candidates; Qwen additionally scores a six-passage, 2,048-token
bundle. Frozen v9 features stay intact. Treatment adds one explicit normalized
bundle-score feature. Matched control/treatment use identical fitting settings.

Results: `outputs/v11/comparison.json` compares treatment vs control, treatment vs
the rerun frozen v9, and control vs frozen v9. Predictions and diagnostics for all
three are saved alongside the caches. Frozen v9 here is rerun on the actual active
corpus, not assumed identical to the historical score.

The restored g32 recipe uses `train-r2.jsonl`, containing all 5,950 train questions;
there are zero untouched calibration questions in that split. See
`analysis/v11/membership-audit.json` (recipe reconstruction, not a checkpoint-bound
membership hash). Calibration therefore uses questions from g32's training set, so results are
explicitly exploratory development comparisons. This setup does not assert a clean
held-out calibration set. Evaluation labels are rejected from fusion fitting.

To test the repaired corpus, first build its store, lexical and dense indexes in
their own directories; invoke `python -m legalir2.v11` with explicit `--store`,
`--lexical`, `--dense` and a fresh `--output`. Use identical paths for train and
validation collection. Do not combine corpus repair and bundle changes when trying
to attribute gains separately.

Manual `gpu-v11.sh` does not upload; the `gpu-v11-auto.sh` wrapper handles uploads.
The public prediction and ZIP creation stage is automatic; competition-site submission
and paid GPU provisioning are not performed.

The code is implemented and CPU-tested, but has not been run end-to-end on a GPU.
Use the benchmark before budgeting time. Both fusion models are evaluated from
one shared validation cache; separate control/treatment GPU passes are unnecessary.
No per-model throughput or rental-cost estimate can be established from the combined
historical rerank timing alone. The richer bundle is an additional Qwen pass, because
the original passage features are retained.
