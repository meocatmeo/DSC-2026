# Resume after the redundant tournament validation solver fails

The run committed 1,150 private questions and then failed at
`fulltext_retrain.validate_rows -> duo.validate_scores -> tournament`. The saved
ability is computed earlier by `fulltext_prior.fit`, whose residual already
passed. `validate_scores` reruns the older Newton solver with normalized prior,
then discards that solution. Its convergence failure does not establish that
the stored fit is invalid. The failed uncommitted matrix is not available in
the saved shard cache, so the specific numerical stall is not reproduced here.

V3 preserves every frozen training/scoring module and the exact v2
`retrain_private_submit.py` bytes, keeping request and shard identities intact.
An additive wrapper replaces only retrained-row validation during execution:
membership, source/evidence digest, unchanged prior, vector sizes/finiteness,
matrix sizes/finiteness/bounds/reciprocity, and the existing fitted-solution
gradient tolerance of 1e-5 are required. The unnecessary second optimization is
omitted. Model inference, `fulltext_prior.fit`, rank calculation and submission
policy remain identical. Invalid/nonstationary rows still fail. A separate
`numerical-validation-repair.json` records wrapper provenance in the output.

On the same running instance, after the failed run completes its final backup:

```bash
(
set -euo pipefail
mkdir -p /workspace/dsc-retrain-private-v3
rclone copy gdrive:DSC-2026-results/releases/tools/20260922-retrain-private-submit-v3 /workspace/dsc-retrain-private-v3 --progress
cd /workspace/dsc-retrain-private-v3
sha256sum -c files.sha256
python3 launch_retrain.py --private-submit
)
```

The same run and existing completed shards resume. No deletion, model download,
environment reinstall or retraining is needed. Up to the failed ten-question
shard is recomputed. Confirm auto-stop re-armed for the current instance; the
watcher still requires successful completion and verified cloud backup.

Tests cover a forced redundant-solver failure with a valid saved optimum,
unchanged predictions, corrupt matrices/stale scores, and a real saved private
shard. No GPU success or private score improvement is claimed by these tests.
