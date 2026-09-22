# Auto-stop for the matched full-text retraining run

This separate helper can be armed from a second terminal on the same Vast GPU.
It does not modify the published training ZIP or interrupt the running process.
The default workspace is `/workspace/DSC-2026-fulltext-retrain-4090-v1` and the
default run is `duo-fulltext-retrain-4090-v1`. Use explicit `--root` and `--run`
arguments if those were changed.

```bash
set -e
mkdir -p /workspace/dsc-retrain-autostop
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-autostop-v1 /workspace/dsc-retrain-autostop --progress
cd /workspace/dsc-retrain-autostop
sha256sum -c vast_stop_retrain.py.sha256
python3 vast_stop_retrain.py --root /workspace/DSC-2026-fulltext-retrain-4090-v1 --run duo-fulltext-retrain-4090-v1
```

The helper detects the instance ID from the Vast environment, or asks for the
**instance ID, not the machine ID**. It uses a provided instance API key or
prompts for a Vast API key with input hidden. Never paste that key into chat.
A read-only Vast request checks authentication and the requested instance ID.
Credentials and watcher logs stay under `~/.config/dsc-vast-stop/`, outside all
DSC workspace backups, with restricted permissions.

Wait for `AUTO-STOP ARMED for instance ...` and confirm the displayed instance ID.
The watcher continues after the terminal closes. Repeating the command does not
create another active watcher for the same instance/run.

The watcher waits through setup/restore for the target retraining exit marker,
then obtains the workspace lock. It requires all of the following before STOP:

- Successful final exit, after the runner has released its backup/workspace lock.
- Matching run inputs, gate and submission-choice records.
- A ZIP matching its recorded SHA256, with every private question and exactly
  five distinct known corpus documents per answer.
- A successful content-hash check of all local run artifacts against
  `gdrive:DSC-2026-results/runs/retrain/<run>/artifacts`, including both models.

A gate rejection with a valid baseline submission is a successful job and can
stop. A training, submission, upload or API failure leaves the instance running;
inspect the printed watcher log and stop it manually if needed. Setup failure
before the target run starts leaves the watcher waiting, without stopping.
The API request is STOP, never destroy: disk and storage charges remain.

The new watcher has five CPU tests covering layout, gate/ZIP mismatches,
backup scoping, lock/exit ordering, backup failure and the API operation.
The original watcher also passes its five tests. The API operation is unchanged
from the previously tested helper. No live GPU has been armed from this local
workspace; the command above must run on the rental instance.
