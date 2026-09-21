# Extend the pilot from 256 to 512 training questions

Use the existing `/workspace/DSC-2026-pilot-4090` workspace on the same RTX 4090.
If the private-submission run is active, let it finish and let its auto-stop
watcher finish before starting this run. Restart the stopped instance as needed.
This additive release does not modify the 256-question pilot or its submissions.

```bash
(
set -euo pipefail
cd /workspace/DSC-2026-pilot-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260920-deep-pilot512-4090-v1
rclone copyto "$REMOTE/pilot512-overlay.zip" pilot512-overlay.zip --progress
rclone copyto "$REMOTE/pilot512-overlay.zip.sha256" pilot512-overlay.zip.sha256
sha256sum -c pilot512-overlay.zip.sha256
unzip -o pilot512-overlay.zip
sha256sum -c --quiet private-files.sha256
sha256sum -c --quiet pilot512-files.sha256
if [ -n "${TMUX:-}" ]; then
  AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot512-4090.sh run
else
  tmux new-session -s pilot512 \
    'AUTO_STOP=1 bash scripts/gpu/gpu-deep-pilot512-4090.sh run'
fi
)
```

Enter the Vast API key when prompted and wait for AUTO-STOP ARMED. Successful
completion and verified backup trigger STOP. Errors leave it running. Storage
charges continue after STOP. This uses the installed environment, not a reinstall.

The fixed SHA256-selected 512 training questions contain the exact original
256 as a prefix. The importer verifies and reuses all 256 saved training raw
scores, including the final one-question shard, and all 429 development raw
scores. It scores only 256 new training questions, fits 23 fusion weights, and
recomputes the development duo comparisons for the new fusion shortlist/prior.
BF16, FlashAttention 2, batch 4 and all neural weights remain unchanged.

The report `training-size-comparison.json` compares the 512 model directly
against the completed 256 model on the same 429 questions. Development is
historically reused; a gain is not proof of a private improvement. There is no
validation or private inference in this run. Its submit stage writes unchanged
baseline ZIPs for the backup/auto-stop workflow, not new candidate answers.

At measured 19.6 s/question, the extra training collection is about **84 minutes**.
Development comparisons, CPU verification/fitting and uploads add time; plan
roughly **2–3 hours** as an estimate, not a benchmark or time limit. Reusing
development raw scores avoids roughly 2.3 hours of redundant collection.

Local output: `outputs/private/private-deep-pilot512-4090-v1/`.
Cloud output: `gdrive:DSC-2026-results/runs/private/private-deep-pilot512-4090-v1/artifacts/`.
Live cloud status is in the sibling `logs/STATUS.txt`. Local log is `run.log`.
Resume using the same launcher with `run`; keep the run ID, batch and assets.
All transfers stay under DSC; AIC and NCKH are not involved.
