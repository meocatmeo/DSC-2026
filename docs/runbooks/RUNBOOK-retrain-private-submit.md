# Direct private submission from the retrained control checkpoint

The user requested submission without the remaining validation. This run uses
the saved rebuilt-evidence `control` checkpoint, frozen private evidence and
candidate pools, and the existing strength 0.25 policy. It does not train, run
development/validation, or select weights. The development screen and original
baseline ZIP remain intact. The new decision explicitly records
`user-requested-unvalidated`, `promotion_gate_passed: false`, and
`validation_evaluated: false`; it does not claim the statistical gate passed.

Restart the same stopped RTX 4090 instance if needed. Its workspace, environment,
models and source comparison caches must still exist. Use its existing rclone
configuration:

```bash
(
set -euo pipefail
mkdir -p /workspace/dsc-retrain-private-submit
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-private-submit-v1 /workspace/dsc-retrain-private-submit --progress
cd /workspace/dsc-retrain-private-submit
sha256sum -c files.sha256
python3 launch_retrain.py --private-submit
)
```

Enter the Vast API key at the hidden terminal prompt if requested. Confirm
`AUTO-STOP ARMED` or `AUTO-STOP ALREADY ARMED`, followed by private inference
progress. The launcher uses tmux and follows the log directly. Ctrl+C closes the
log view only. Rerun the same command to follow/resume. It copies and checks the
saved control model into the separate output directory so its model, cache and
submission are all backed up together. It needs 30 GiB free under the existing
launcher resume check. The old completed training watcher, if still waiting, is
retired under the workspace lock before the new watcher is armed.

Local output:
`outputs/retrain/duo-fulltext-retrain-private-control-4090-v1/submission-private-duo.zip`

Cloud output:
`gdrive:DSC-2026-results/runs/retrain/duo-fulltext-retrain-private-control-4090-v1/artifacts/submission-private-duo.zip`

Progress logs are in the same run's `run.log` locally and its `logs/` cloud
directory. Every completed shard is resumable. Successful completion, a verified
complete ZIP, and verified cloud backup permit automatic STOP. A failure leaves
the instance running. STOP preserves disk, so storage charges continue.

This generates a candidate ZIP for manual competition upload. Private scoring
requires 2,080 questions; use the initial progress lines for a measured ETA.
There is no new private score yet. CPU tests cover stage order, failure backups,
source preservation, explicit override metadata, launcher dispatch, and the
existing auto-stop checks. They are not a live GPU or Vast API integration test.
