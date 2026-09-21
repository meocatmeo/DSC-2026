# Private submission from the completed pilot

Restart the same Vast instance if auto-stop has stopped it. Use its existing
`/workspace/DSC-2026-pilot-4090` directory and configured rclone. This additive
release preserves the pilot and its published manifest. Run it after the pilot
has exited and its auto-stop watcher has finished; do not race two GPU runs.

```bash
(
set -euo pipefail
cd /workspace/DSC-2026-pilot-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-deep-pilot-submit-4090-v1
rclone copyto "$REMOTE/pilot-submit-overlay.zip" pilot-submit-overlay.zip --progress
rclone copyto "$REMOTE/pilot-submit-overlay.zip.sha256" pilot-submit-overlay.zip.sha256
sha256sum -c pilot-submit-overlay.zip.sha256
unzip -o pilot-submit-overlay.zip
sha256sum -c --quiet private-files.sha256
sha256sum -c --quiet pilot-submit-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot-submit-4090.sh run
else
  tmux new-session -s deep-submit4090 \
    'AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot-submit-4090.sh run'
fi
)
```

Enter the Vast API key if prompted and wait for AUTO-STOP ARMED. The existing
watcher requests STOP only after successful completion and backup verification.
Errors leave the instance running; storage charges continue after STOP.

The launcher restores models and source data as needed, copies the completed
pilot checkpoints into a separate run, verifies them, then runs private scoring
and new duo comparisons for all 2,080 questions. It prefers local pilot caches;
if the local calibration is missing, it downloads them from the DSC remote.
Restored training and development caches are checked on CPU, not recomputed on
GPU. No refitting, recalibration or validation inference is performed.

Exact frozen pilot: 256 training questions, development recall 0.9502719503 →
0.9514374514, fusion digest
`7a2cf9fba9b857e9ec2834ab80146ddc024530955bf19321ceaed510f046879c`.
The continuation refuses different inputs, fusion weights or development scores.
The new report explicitly records development-only selection and skipped
validation; no private score is known. BF16, batch 4, FlashAttention 2 remain.

Local outputs:
`outputs/private/private-deep-pilot-submit-4090-v1/`.
Cloud outputs:
`gdrive:DSC-2026-results/runs/private/private-deep-pilot-submit-4090-v1/artifacts/`.
Live status is in the sibling `logs/STATUS.txt`; local log is `run.log`.

`submission-private-duo.zip` contains the NEW candidate answers.
`submission-private-v9.zip` retains the original V9 answers.
`submission-choice.json` records how many answer sets changed.
The launcher generates files; it does not upload them to the competition site.

At the measured training collection speed of 19.6 s/question, private raw
scoring alone would take about 11.3 hours. Private duo comparisons, CPU checks
and uploads add time. This is not an end-to-end forecast: read the stage ETAs.
Resume after a failure with the same `run` command and run ID to reuse shards.

Download the new duo ZIP after completion on any configured machine:

```bash
rclone copyto \
  gdrive:DSC-2026-results/runs/private/private-deep-pilot-submit-4090-v1/artifacts/submission-private-duo.zip \
  submission-private-duo-deep.zip --progress
```
