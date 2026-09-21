# Private submission using the completed 512-question model

Use the instance that completed the 512 pilot and its existing workspace. If
auto-stop has stopped it, restart it first. Keep the separate 256-model private
inference running if desired: these runs write to different remote folders.

```bash
(
set -euo pipefail
cd /workspace/DSC-2026-pilot-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260921-deep-pilot512-submit-4090-v1
rclone copyto "$REMOTE/pilot512-submit-overlay.zip" pilot512-submit-overlay.zip --progress
rclone copyto "$REMOTE/pilot512-submit-overlay.zip.sha256" pilot512-submit-overlay.zip.sha256
sha256sum -c pilot512-submit-overlay.zip.sha256
unzip -o pilot512-submit-overlay.zip
sha256sum -c --quiet private-files.sha256
sha256sum -c --quiet pilot512-files.sha256
sha256sum -c --quiet pilot512-submit-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot512-submit-4090.sh run
else
  tmux new-session -s deep512-submit \
    'AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot512-submit-4090.sh run'
fi
)
```

Enter the Vast API key when prompted; confirm AUTO-STOP ARMED identifies this
instance. STOP follows successful completion and verified backup. Failure leaves
the instance running; storage charges continue after STOP.

This run pins the exact saved 512 model with development recall 0.9520202020,
fusion hash `84787ad63ad219de43be5b28b80ec32556105da5a186aa1c3099b8cc694285d7`.
It does not retrain, recalibrate, rerun development, or run validation. Its report
explicitly records development-only selection and unknown private performance.

The launcher snapshots completed raw private shards uploaded by
`private-deep-pilot-submit-4090-v1` into `outputs/reuse/deep-pilot256-private`.
It verifies identical retrieval/model/evidence/settings and replays every raw
feature vector against the frozen source before importing those shards. Fusion
weights do not affect raw reranker scores, so they can be reused. Missing raw
shards are computed normally. Valid existing 512 raw shards are preserved on
resume. It does not reuse the 256 model's rankings or comparison matrices: duo
comparisons run using the new 512 shortlist and prior.

At packaging time, 328 five-question shards (about 1,640/2,080 questions) were
available, approximately 2.70 GB. The launcher takes a fresh snapshot on every
`run`; more may be available by launch. The saved transfer has a 30-minute
timeout and prints progress. A snapshot can be partial while the other GPU runs.
This is not live synchronization during inference. `private-reuse.json` reports
the import count. Do not infer a whole-run ETA from remaining raw scoring alone:
2,080 duo comparisons, verification and backups still follow.

Local outputs: `outputs/private/private-deep-pilot512-submit-4090-v1/`.
Cloud outputs: `gdrive:DSC-2026-results/runs/private/private-deep-pilot512-submit-4090-v1/artifacts/`.
Local log: `run.log`; live cloud status is in the sibling `logs/STATUS.txt`.

`submission-private-duo.zip` contains NEW 512-model predictions.
`submission-private-v9.zip` remains the original V9 submission.
`submission-choice.json` reports changed answer sets and skipped validation.
The runner creates the files; it does not submit them to the competition site.

Resume using the same launcher with `run`. Do not run two writers using this
512-submission run ID simultaneously. Only DSC remote paths are touched.
