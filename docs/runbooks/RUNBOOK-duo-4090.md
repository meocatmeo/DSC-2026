# Fresh RTX 4090: full-text duo inference

Use a Linux x86_64 **CUDA 12.8 development image** with `nvcc`, Python 3.10–3.12, one RTX 4090 with 24 GB VRAM, at least 100 GB allocated disk and 60 GiB free. Prefer at least 32 GB system RAM and 8 CPU cores for indexing and installation. The launcher limits FlashAttention compilation to two jobs by default.

This runs the automated full-text evidence policy using the saved comparator, original calibration and supplied corpus. It does not include the manual answer replacements. No retraining is required. Hardware support is separate from competition eligibility; this guide does not certify eligibility.

## Install rclone and configure access

On an Ubuntu/Debian GPU terminal:

```bash
set -euo pipefail
SUDO=()
if [ "$(id -u)" -ne 0 ]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y curl ca-certificates unzip tmux \
  python3 python3-venv python3-dev build-essential git util-linux
curl -fsSL https://rclone.org/install.sh -o /tmp/install-rclone.sh
"${SUDO[@]}" bash /tmp/install-rclone.sh
mkdir -p ~/.config/rclone
chmod 700 ~/.config/rclone
```

Upload the existing local `D:\DSC-2026\.secrets\rclone.conf` through the rental file browser to `/workspace/rclone.conf`, then:

```bash
install -m 600 /workspace/rclone.conf ~/.config/rclone/rclone.conf
rclone lsd gdrive:DSC-2026-results
nvidia-smi
nvcc --version
```

## Download and run

```bash
mkdir -p /workspace/DSC-2026-4090
cd /workspace/DSC-2026-4090
REMOTE=gdrive:DSC-2026-results/releases/private/20260919-duo-fulltext-v2-4090
rclone copyto "$REMOTE/private-inference-overlay.zip" private-inference-overlay.zip --progress
rclone copyto "$REMOTE/private-inference-overlay.zip.sha256" private-inference-overlay.zip.sha256
sha256sum -c private-inference-overlay.zip.sha256
unzip -o private-inference-overlay.zip
sha256sum -c --quiet private-files.sha256
tmux new -s duo4090
```

Inside tmux:

```bash
cd /workspace/DSC-2026-4090
bash scripts/gpu/gpu-duo-4090.sh all
```

Detach with Ctrl+B then D. Reconnect with `tmux attach -t duo4090`.

`all` installs the pinned Torch/packages and FlashAttention 2.8.3, checks the GPU and runs a BF16 FlashAttention kernel, restores the models/indexes and saved V9 cache, builds the full-text side index, runs duo, and creates both ZIPs. `submit` creates files only; no competition-site upload occurs. Logs and outputs back up automatically through the shared DSC rclone worker.

The 4090 profile uses comparator batch **4**, compared with the H100 default 8. It keeps BF16, FlashAttention2, the 2,048-token context, all model weights and candidate/passage selection unchanged. The V9 cache is restored from `private-v9-duo-9da4e0cb-v1` and validated against source assets/questions. Old duo shards are not imported. The shared Python environment is still named `.venv-h100`; this name does not restrict its GPU.

FlashAttention2 supports Ada/RTX 4090: [upstream 2.8.3 documentation](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#nvidia-cuda-support). The setup performs a real kernel check on the rental host. This release has CPU regression and launcher tests, but has not been benchmarked end to end on a real 4090. Neither identical predictions across GPU architectures nor a score increase is guaranteed.

## Resume and output

After environment installation, resume with:

```bash
bash scripts/gpu/gpu-duo-4090.sh run
```

Use the same GPU, code, inputs and batch size for the same run ID. If batch 4 runs out of GPU memory, stop the previous run and start a separate run at batch 2:

```bash
PRIVATE_RUN_ID=private-duo-fulltext-v2-4090-b2 INFER_BATCH=2 \
  bash scripts/gpu/gpu-duo-4090.sh run
```

It reuses the restored assets and V9 cache, but recomputes duo to avoid mixing batching identities. Set `V9_SOURCE_RUN_ID=''` only if you deliberately want to recompute V9; its saved SDPA/batch-8 settings remain unchanged. A missing or incompatible requested cache stops with an error.

Default results:

```text
outputs/private/private-duo-fulltext-v2-4090/submission-private-duo.zip
outputs/private/private-duo-fulltext-v2-4090/submission-private-v9.zip
gdrive:DSC-2026-results/runs/private/private-duo-fulltext-v2-4090/artifacts/
```

From Windows PowerShell, download the new duo result:

```powershell
Set-Location D:\DSC-2026
.\tools\bin\rclone.exe --config .\.secrets\rclone.conf copyto `
  gdrive:DSC-2026-results/runs/private/private-duo-fulltext-v2-4090/artifacts/submission-private-duo.zip `
  .\submissions\private\submission-private-duo-fulltext-4090.zip --progress
```

Build this release locally with `python scripts/package/package-private.py --fulltext --gpu 4090`. All shell scripts remain ignored by Git and are included in the release package. Only DSC cloud paths are used.
