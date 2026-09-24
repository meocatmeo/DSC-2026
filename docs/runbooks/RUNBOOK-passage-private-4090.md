# Private passage-order submission (0.97 primary, 0.98 alternative)

User requested a speculative submission after the 429-question development run
tied: zero recall wins/losses, 106 changed sets under the 0.90 policy. Replaying
the completed GPU scores under the current best 0.98 policy also ties, with 91
changed sets. This does not demonstrate a private gain. User subsequently reported
0.99 at 0.9520; the user then reported 0.97 at 0.9533, which is now best.

This release changes passage ordering, using the same neural checkpoints, original
candidate pools, prior values and original packed tokens. It scores all 2,080
private questions with both comparators, applies fixed 0.97 and 0.98 policies to those same scores, and
packs `submission-private-passage-order.zip` (0.97) and
`submission-private-passage-order-098.zip` (0.98). No extra GPU scoring is needed
for the alternative. No training, validation, new candidate
retrieval, quantization, threshold search or automatic competition upload occurs.

## Start on the previous or a fresh rental

Use RTX 4090 24 GB, CUDA 12.8 devel, Python 3.12. A fresh rental needs the existing
pinned environment installer and at least 60 GiB free disk. Configure rclone
`gdrive:` first. Restart a stopped rental through Vast before running commands.

Download `start.sh` from
`gdrive:DSC-2026-results/releases/passage-order/20260923-passage-private-4090-v2`
and verify its published SHA256. Then run `bash /tmp/dsc-passage-private-start.sh`.
The start script pins the ZIP digest and checks every extracted file.

The v2 bootstrap supports a fresh rental after destruction of the old instance.
It checks build tools even when tmux is preinstalled, checks Python/CUDA before
launch, and completes a partial local environment install when `all` is rerun.
The watcher ignores completion markers from before this launch. Model scores,
prepared inputs, policies, workspace and cloud run ID are unchanged from v1;
this is a launch fix, not a different experiment.

Workspace: `/workspace/DSC-passage-order-private-4090-v1`.
Tmux: `dsc-passage-private`; attach with `tmux attach -t dsc-passage-private`
(or `tmux switch-client -t dsc-passage-private` from inside tmux).

When the old development workspace is present, the bootstrap requires it to have
exited. It retires only that workspace's known development auto-stop process so
it cannot stop the new job. It reuses the previous environment and model paths;
the original code/output files are not edited. A fresh rental downloads the two
model checkpoints and installs the original dependencies including FlashAttention.

Stages: restore models and prepared inputs → check → two-question smoke run →
score original-duo → score retrained-control → submit → backup. Each model first
replays an unchanged question to check runtime compatibility. Unchanged edges
are reused; both orientations of changed edges are rescored. Every completed
question is written atomically. The runner streams prepared questions to limit
CPU RAM consumption. Resume with:

```bash
cd /workspace/DSC-passage-order-private-4090-v1
PASSAGE_PY=/workspace/DSC-passage-order-4090-v2/.venv-h100/bin/python \
PASSAGE_MODELS=/workspace/DSC-passage-order-4090-v2/models-passage-order \
bash scripts/gpu/gpu-passage-private-4090.sh run
```

Omit the environment overrides on a fresh rental with a run-local environment.

## Submission and backups

Run ID: `passage-order-private-4090-v1`.
Local outputs: `outputs/passage-order/passage-order-private-4090-v1/`.
Cloud root:
`gdrive:DSC-2026-results/runs/passage-order/passage-order-private-4090-v1`.
Logs upload every 60 seconds under `logs/`; committed scores, choice receipt,
changes and submission ZIP upload under `artifacts/`. Partial predictions cannot
produce a ZIP: submit checks all 2,080 questions, both complete score caches,
model replay checks, known IDs, five unique answers and exact old-best replay.

Retrieve the experimental submission:

```bash
rclone copyto gdrive:DSC-2026-results/runs/passage-order/passage-order-private-4090-v1/artifacts/submission-private-passage-order.zip submission-private-passage-order.zip --progress
```

## Auto-stop

After launching the job, the bootstrap runs `vast_stop_passage_private.py` in the
interactive terminal. It detects the instance or prompts for its ID; API key is
read from the existing environment or a hidden prompt. Do not put keys in chat.
Wait for `AUTO-STOP ARMED`. Arming/authentication failure leaves the GPU job running;
rerun the helper after correcting credentials. No claim of arming is made by the
local packager. Prior development watcher authorization does not automatically
arm this new watcher.

The watcher requires successful exit, the pinned private plan, all prepared inputs,
all 4,160 completed score files, replay receipts, a valid final ZIP, and a successful
hash-based rclone check before requesting STOP. Failure leaves the instance running.
STOP preserves disk; storage charges continue. It never requests destroy.

## Estimated time and provenance

Budget roughly 8–12 hours for two private scoring passes from the earlier 4090
throughput, plus setup/download. The logs report ETA for the **current model only**;
the second comparator still runs afterward. Actual speed may differ.

Preparation verifies private raw score rows against the saved deep comparison
digest, the original baseline and retrained shard hashes, exact old evidence and
token reconstruction, and replay of all 0.9525 and 0.9533 predictions. Deep-pool priors and
new documents are not used. Source code and plan are frozen. GPU scoring and new
private score remain unverified until this release runs on the rental.
