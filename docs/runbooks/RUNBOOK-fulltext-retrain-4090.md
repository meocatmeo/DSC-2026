# Matched duo evidence training on RTX 4090

This is a controlled training experiment, not an established score improvement.
Use a Linux x86_64 CUDA **12.8 devel** image, Python **3.10–3.12**, one RTX 4090
24 GB, at least **80 GiB free disk** and preferably 32 GB or more system RAM.
BF16 and FlashAttention 2 are required. No quantization or causal-conv1d is used.

The release is `releases/retrain/20260921-fulltext-retrain-4090-v1` under
`gdrive:DSC-2026-results`. The downloadable `start.sh` installs rclone when needed,
accepts your pasted configuration through the terminal, checks the release ZIP,
extracts to a new workspace, and starts one named tmux session. It does not nest
tmux: an existing client switches to that session.

After rclone is configured, download and run:

```bash
rclone copyto gdrive:DSC-2026-results/releases/retrain/20260921-fulltext-retrain-4090-v1/start.sh /tmp/dsc-fulltext-retrain-start.sh
bash /tmp/dsc-fulltext-retrain-start.sh
```

For a fresh instance without rclone, the complete terminal block is:

```bash
set -e
curl -fsSL https://rclone.org/install.sh | bash
umask 077
mkdir -p "$HOME/.config/rclone"
echo 'Paste the complete rclone.conf, then press Ctrl+D:'
cat > "$HOME/.config/rclone/rclone.conf"
rclone copyto gdrive:DSC-2026-results/releases/retrain/20260921-fulltext-retrain-4090-v1/start.sh /tmp/dsc-fulltext-retrain-start.sh
bash /tmp/dsc-fulltext-retrain-start.sh
```

Paste the configuration only after the `cat` prompt appears. The configuration
stays outside the workspace and is never included in backups or releases.
Use a root terminal on the rental instance for rclone/system package installation.

## What the default run does

1. Installs pinned dependencies and tests a FlashAttention CUDA kernel.
2. Restores frozen models, indexes and the exact scored full-text private cache.
3. Checks the verified v3 archives and the 429/1050 external evaluation exclusions.
4. Trains **old-evidence-control**, then **control**, from the same original Qwen
   partner weights. Both have 1,427 fit questions, 131 monitor questions, identical
   ordered pairs, seed 2026, 2 epochs, microbatch 2, accumulation 8, learning rate
   2e-5, LoRA rank 16 and maximum pair length 2,048. The monitor chooses the best
   epoch by pair loss; it never authorizes deployment.
5. Scores both checkpoints on all 429 development questions, using identical
   saved inference token sequences, candidates, priors and fixed blend strength
   0.25. Reuses the 429 verified frozen comparator baseline rows bundled in the
   release; those comparisons do not consume another GPU pass.
6. Continues to validation only if the rebuilt-evidence arm has positive recall
   deltas against both the matched old-evidence arm and the frozen baseline.
   This is a predeclared cost screen, **not a promotion decision**.
7. If screened in, scores the same fixed policies on all 1,050 validation
   questions. Promotion requires pooled baseline delta at least **+0.0052** and
   a paired bootstrap 95% interval entirely above zero, plus an entirely positive
   paired interval against the matched old-evidence arm. The latter guards the
   claim that the evidence rebuild itself helped.
8. Runs private inference only if that gate passes. `submit` always runs and
   packages either the admitted model's answers or the unchanged scored baseline.
   It does **not** upload answers to the competition website.

`hard-negatives.json.gz` is included as a third, optional arm. The default run
does not train it or choose it after inspecting validation. `train-hard-negatives`
only produces an exploratory checkpoint; it is not eligible for this release
gate. A comparison against `control` would isolate added negatives and needs its
own predeclared evaluation before deployment.

The matched comparison isolates evidence as the training intervention under the
same selection procedure. It does not establish universal superiority of the
new passages. The audit found some worse packed excerpts. The 131-question monitor
is not a fresh holdout relative to historical models. All external labels have
also been used historically. Bootstrap intervals describe paired sampling
uncertainty; they do not erase historical selection bias or guarantee private gain.
There is no universal fixed resolution for 131, 429 or 1,479 questions: precision
depends on the actual paired per-question differences.

## Resume, logs and outputs

Workspace: `/workspace/DSC-2026-fulltext-retrain-4090-v1`.
Default run ID: `duo-fulltext-retrain-4090-v1`.

```bash
cd /workspace/DSC-2026-fulltext-retrain-4090-v1
bash scripts/gpu/gpu-fulltext-retrain-4090.sh run
```

`run` restores required assets and resumes without reinstalling packages. Both
training arms have separate model/checkpoint directories. Training resumes from
the last saved optimizer step (every 100 updates and each epoch). Scoring resumes
verified ten-question shards. Changing data, code or batch identity needs a new
run ID. Do not unzip this release over a running worker or the frozen workspace.

Local log:
`outputs/retrain/duo-fulltext-retrain-4090-v1/run.log`.

Cloud root:
`gdrive:DSC-2026-results/runs/retrain/duo-fulltext-retrain-4090-v1/`.
Live logs/status are under `logs/`; models, checkpoints, reports and the submission
are under `artifacts/`. Background backup runs every 60 seconds, with additional
stage/final backups. Transfers are bounded; failures preserve local files.
A failed required backup stops the stage progression. Resume once connectivity
returns. No AIC or NCKH paths are read or written.

Inspect `screen.json`, `gate.json` and especially `submission-choice.json`.
Only `candidate_deployed: true` means the submitted predictions use the new model.
The one deployed comparator replaces the old one; training two arms does not
add them together as voters. Each merged model gets a parameter-budget audit.

The launcher does **not** arm Vast auto-stop. The old private-run watcher expects
a different output layout; do not assume it watches this experiment. Stop the
instance after successful backup if you are handling billing manually.

## Runtime expectations and verification

This costs more than a single retraining: two matched training arms and two new
comparator passes on development (the frozen baseline is cached). If the development screen fails, validation/private
GPU work is skipped. If it passes, validation and up to 2,080 private questions
follow. Do not interpret a training or development ETA as the total job ETA.
There is no measured 4090 training ETA for this release yet; use the logged
seconds/update and seconds/question once it starts. This is an hours-long
experiment, potentially an overnight run when all gates pass.

CPU validation covers archive hashes, identical pairs/splits, absence of external
IDs and normalized aliases, gate behavior, cache provenance and release contents.
Bash syntax is checked. Actual 4090 training and CUDA memory/runtime behavior must
still be verified on the rental GPU; startup checks fail rather than silently
changing precision, attention or batch settings.
