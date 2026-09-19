# Private V9 + duo inference

**Layout-v2:** launchers are under `scripts/gpu/`; cloud model backups are under
`assets/models/` and runs under `runs/`. Older downloaded launchers use retired
paths. Refresh using the existing-workspace commands below before resuming.

This run produces two independent answers for every one of the 2,080 private questions:
`submission-private-v9.zip` and `submission-private-duo.zip`. Each ZIP contains exactly
`submission.json`, with five distinct corpus document IDs per question.

It uses the saved V9 stacker and the trained `duo-20260918T110105Z` comparator at its
development-selected strength **0.25**. It does not train, calibrate, run validation,
apply manual public-ID edits, or use private labels. The user's request authorizes both
outputs even though the original duo selection report was statistically inconclusive.
That report and the trained model signatures remain unchanged.

## Fresh rental: complete setup

Use one H100, a **Linux x86_64 CUDA 12.8 development image** (with `nvcc`),
Python 3.10-3.12 (3.12 matches the old host), and at least 100 GB allocated disk
with 60 GiB free. A CUDA runtime-only image cannot build FlashAttention.
The host driver must support CUDA 12.8 or newer; the toolkit inside the image
should be 12.8. The script checks these before inference.

On the GPU terminal (Ubuntu/Debian image):

```bash
set -euo pipefail
SUDO=()
if [ "$(id -u)" -ne 0 ]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y curl ca-certificates unzip tmux nano \
  python3 python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh -o /tmp/install-rclone.sh
"${SUDO[@]}" bash /tmp/install-rclone.sh
mkdir -p ~/.config/rclone
chmod 700 ~/.config/rclone
nano ~/.config/rclone/rclone.conf
```

Paste the contents of your local `D:\DSC-2026\.secrets\rclone.conf`, save with
Ctrl+O, Enter, and exit with Ctrl+X. Do not paste the configuration into chat.
Alternatively upload that file through the rental file browser to
`/workspace/rclone.conf`, then use:

```bash
install -m 600 /workspace/rclone.conf ~/.config/rclone/rclone.conf
```

Continue on the GPU terminal:

```bash
chmod 600 ~/.config/rclone/rclone.conf
rclone lsd gdrive:DSC-2026-results
nvidia-smi
nvcc --version
mkdir -p /workspace/DSC-2026-h100
cd /workspace/DSC-2026-h100
rclone copyto gdrive:DSC-2026-results/releases/private/20260919-layout-v2/private-inference-overlay.zip private-inference-overlay.zip --progress
rclone copyto gdrive:DSC-2026-results/releases/private/20260919-layout-v2/private-inference-overlay.zip.sha256 private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
tmux new -s private
```

Inside tmux, run:

```bash
cd /workspace/DSC-2026-h100
bash scripts/gpu/gpu-private-setup.sh all
```

The setup installs the recorded package versions in `.venv-h100`, installs required
FlashAttention 2.8.3, runs a CUDA kernel check, then invokes the existing restore,
preflight, V9, duo and submit stages. No retraining or validation is run.
Causal-conv1d is not needed. Failed installs/kernel checks stop the run; there is
no silent attention fallback. Setup has the same cloud logging and shutdown
handling as inference, under `gdrive:DSC-2026-results/runs/setup/private/<attempt>/`.

Detach with Ctrl+B, then D; reconnect with `tmux attach -t private`. After an
interruption, rerun `bash scripts/gpu/gpu-private-setup.sh all`. Completed private shards resume.
Environment installation is idempotent; a new setup log directory is used per attempt.

If the cloud `assets/models/partner_qwen` backup is absent, restore downloads the public
`Qwen/Qwen3-Reranker-0.6B` at commit `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
Every downloaded file must match its SHA256 in the original duo source signature
before it replaces a local file. The verified model is then backed up to rclone.
The fine-tuned embedding and g32 models still require their exact saved backups.

## Existing H100 workspace

The existing compatible `.venv-h100` environment, CUDA/BF16 GPU, and configured `gdrive:`
rclone remote are required. Models and exact index backups are restored automatically.
The overlay includes the private questions, inference code, frozen V9 configuration,
tests, and logging scripts; it contains no credentials or model weights.

```bash
cd /workspace/DSC-2026-h100
rclone copyto gdrive:DSC-2026-results/releases/private/20260919-layout-v2/private-inference-overlay.zip private-inference-overlay.zip --progress
rclone copyto gdrive:DSC-2026-results/releases/private/20260919-layout-v2/private-inference-overlay.zip.sha256 private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
bash scripts/gpu/gpu-private-h100.sh all
```

Run in tmux if SSH disconnects are possible. Repeat the last command after an interruption
to resume committed private shards. The V9 ZIP is saved and uploaded before duo begins.

The stages are `restore`, `preflight`, `v9`, `duo`, `submit`, and `all`. `submit` verifies
and packages both complete caches without scoring questions. `duo` requires the completed
private V9 cache. No public query cache is restored. `all` runs through `submit`.

V9 uses the saved SDPA/batch-8 runtime; duo uses its saved FlashAttention2 backend.
Do not silently change backends or upgrade packages. Preflight checks the recorded
Torch, Transformers, tokenizer, FlashAttention and other package versions, the model
weights, index bytes and scoring code. `flash-attn==2.8.3` is already required by the
saved comparator environment. Causal-conv1d is not needed by these Qwen3 models.

The inference launcher targets an **existing environment**. On a new GPU host,
use `scripts/gpu/gpu-private-setup.sh all` above to install it first. Setting `PY` when calling
the inference launcher selects another existing environment but does not bypass
version checks.

## Output and logs

Default local directory:
`outputs/private/private-v9-duo-9da4e0cb-v1/`

Default cloud directory:
`gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/`

```bash
rclone cat gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/logs/STATUS.txt
rclone copyto gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/artifacts/submission-private-v9.zip submission-private-v9.zip --progress
rclone copyto gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/artifacts/submission-private-duo.zip submission-private-duo.zip --progress
```

Logs and checkpoints upload every 60 seconds and on stage completion/exit. A failed
final upload returns a nonzero exit status and retains local artifacts. No competition
website submission is performed automatically.

Download both results to your Windows PC in PowerShell:

```powershell
Set-Location D:\DSC-2026
.\tools\bin\rclone.exe --config .\.secrets\rclone.conf copyto gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/artifacts/submission-private-v9.zip .\submission-private-v9.zip --progress
.\tools\bin\rclone.exe --config .\.secrets\rclone.conf copyto gdrive:DSC-2026-results/runs/private/private-v9-duo-9da4e0cb-v1/artifacts/submission-private-duo.zip .\submission-private-duo.zip --progress
```

The scorers print seconds per question and a stage ETA after each ten-question shard.
Use those measurements for the new host; no GPU runtime has been measured for this
private input locally. There is no retraining time, and V9 retrieval/reranking runs once.

## Identity and resume

The private question file SHA256 is
`9da4e0cb84204fed924251c35744c93879556e67a440332015ea3b62f3c355bc`.
Its `answer: null` values are normalized only in an in-memory scoring copy, preventing
the earlier NoneType iteration failure without changing the official input.

Private resume markers bind question content/order, models, code, backend, batches,
source assets and calibration. Duo additionally binds the exact V9 shard hashes.
Corrupt or wrong-question shards fail closed. Use a new `PRIVATE_RUN_ID` for deliberately
changed inputs; do not replace the original public file or edit old training signatures.

Restore uses `--ignore-existing` to preserve local completed assets. If preflight reports
a corrupt existing asset, restore that named file from its exact backup; the runner
does not rebuild an index or overwrite mismatching weights to hide an identity error.

CPU tests cover null-label handling, changed-input rejection, interrupted-run reuse,
V9-to-duo cache sharing, saved comparator attention, asset hashes, ZIP coverage, and
shell syntax/help. Real H100 inference must still run to generate the actual answers.
