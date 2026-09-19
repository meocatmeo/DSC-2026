# H100 ranking experiment and submission

Extract `analysis/ranking-h100/ranking-h100-ready.zip` on the rental machine. It includes the code, tests, original question splits, V9 baseline results and the exact frozen corpus. Use the contents together; the current `artifacts2` corpus is different from V9. No model weights or rclone credentials are included.

## Start

Use a Linux H100 with a CUDA 12.8 **development** image, Python 3.11 with `venv`, `g++`, `nvcc`, `rclone`, and the usual GNU utilities (`flock`, `setsid`, `timeout`, `sha256sum`). Allow at least 40 GiB free disk at launch; 80+ GiB disk and 64 GiB host RAM are practical targets for model copies, indexes and compilation. Configure the existing `gdrive:` rclone remote using your own credentials before launching. The remote must allow writes and contain the trained checkpoints:

```text
gdrive:DSC-2026-results/assets/models/embedding/
gdrive:DSC-2026-results/assets/models/reranker-g32/
gdrive:DSC-2026-results/assets/models/partner_qwen/   # optional: official download fallback
```

From the extracted directory, inside `tmux`:

```bash
tmux new -s ranking-h100
bash scripts/gpu/gpu-ranking-h100.sh
```

The default executes **env → restore → indexes → preflight → benchmark → run → submit**. `submit` generates public predictions, validates question coverage and document IDs, packs exactly `submission.json` inside `submission.zip`, and uploads the artifacts through rclone. Submit that ZIP to the competition yourself; this launcher does not operate the competition website.

The shell follows the existing `scripts/gpu/gpu-v11-auto.sh` stage functions and companion runtime layout: strict Bash errors, one workspace lock, run IDs, offline local inference, explicit checkpoint restores, heartbeat uploads and final backup on exit. All stages stop on a required failure.

## What runs

1. Verify the release hashes and prove rclone log upload works before installation or GPU inference.
2. Create a separate `.venv-h100` with PyTorch 2.7.1/cu128, Transformers 4.51.3 and NumPy 2.2.6. The existing `.venv` is untouched. Restore the trained embedding and g32 reranker; an incomplete trained checkpoint stops the run.
3. Build an isolated store and lexical index from the bundled V9 corpus. Restore a matching dense index or rebuild it. Verify corpus, chunk IDs, model identity and parameter counts before scoring.
4. Try FlashAttention 2.8.3, bounded to 20 minutes with four compiler jobs by default. Test its CUDA kernel. Benchmark SDPA batches 8/32 and, when available, FlashAttention batches 16/32 on six **training** questions. Adopt a faster setting only if it preserves the reference top-five predictions and improves measured elapsed time by over 10%. Otherwise retain SDPA/batch 8. The full baseline replay then checks for broader drift.
5. Replay frozen V9 on validation. Stop before calibration if recall drifts by more than 0.002 absolute or candidate recall@50 by more than 0.001.
6. Collect the first 2,000 calibration records, excluding validation IDs/text and duplicate text. Compare the frozen V9 stacker, a matched refit with old features, a refit with corrected missing ranks, and a refit with corrected ranks plus better document passages. Both rerankers still see all 50 candidates, with three passages each.
7. Promote a treatment only when its paired bootstrap 95% interval has a positive lower bound against **both** the frozen V9 and the matched refit control. Otherwise retain the original V9 public answers. This is an exploratory gate on an already reused validation set, not a guarantee of an improvement over the public 0.9508 score.
8. Run **submit**, write the ZIP and upload results. The selected treatment's public GPU pass is resumable too.

External FlashAttention is enabled only for Qwen's causal head. XLM-R continues using PyTorch SDPA. `causal-conv1d` is available with `INSTALL_CAUSAL_CONV1D=1`, but none of these three checkpoints calls it, so installing it provides no expected speedup. This setup follows the [PyTorch CUDA wheel instructions](https://pytorch.org/get-started/previous-versions/) and [FlashAttention 2.8.3 installation requirements](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3). It uses the Transformers-compatible FlashAttention 2 integration.

## Resume, logs and submit

The first launch saves its run ID in `outputs/.ranking-h100-active`. Repeating the same command resumes committed 25-question shards. At most the current shard needs rescoring after interruption. Corrupt shards are rescored; changed code, files, model hashes, package versions or runtime settings are rejected under the same ID. Keep the release and environment unchanged to resume.

```bash
# Same machine: finish all remaining stages, including submit.
bash scripts/gpu/gpu-ranking-h100.sh

# Generate/package/upload the submission after run has completed.
bash scripts/gpu/gpu-ranking-h100.sh submit

# Fresh machine: same release, remote and run ID; restore and resume all stages.
RUN_ID=20260918T120000Z bash scripts/gpu/gpu-ranking-h100.sh

# Separate experiment with different settings; choose a new ID.
RUN_ID=rank-passages-02 LIMIT=3000 bash scripts/gpu/gpu-ranking-h100.sh
```

Stages available: `env`, `restore`, `indexes`, `preflight`, `benchmark`, `run`, `submit`, `all`. Individual later stages expect earlier inputs to exist; `all` is the recovery entry point. The standalone `submit` stage rechecks the environment and input hashes before using the saved selection and weights.

Every heartbeat (default 60 seconds, plus transfer time) uploads a frozen log snapshot, status and completed artifacts. Transfers use checksums, timeouts and bounded retries. A failed heartbeat retries later; a failed final upload returns a nonzero exit status and keeps local files. Shutdown preserves the original failure code. An abrupt machine loss preserves only the shards already uploaded. Monitor:

```bash
rclone cat gdrive:DSC-2026-results/runs/ranking-h100/<RUN_ID>/logs/STATUS.txt
```

Results are under `outputs/ranking-h100/<RUN_ID>/` and:

```text
gdrive:DSC-2026-results/runs/ranking-h100/<RUN_ID>/
  logs/STATUS.txt
  logs/<ATTEMPT_ID>/run.log
  artifacts/baseline-check.json
  artifacts/comparison.json
  artifacts/selection.json
  artifacts/stacker-{control,ranks,document}.json
  artifacts/submission.json
  artifacts/submission.zip
```

Other options: `BASE_RESULTS` changes the remote base; `FLASH_ATTENTION=off` disables the optional build; `MAX_JOBS`, `NVCC_THREADS`, `FLASH_BUILD_TIMEOUT`, `LOG_EVERY`, and `SHARD_SIZE` control resource use. Preserve scoring and package options when resuming an existing ID. The exact store, lexical and dense index files are backed up under the run's `indexes/` directory for cross-machine recovery; rebuilding equivalent indexes would change their byte hashes and invalidate old checkpoints. Dense indexes are additionally cached under `ranking-h100/dense/<store-sha256>/`. Interrupted dense builds use a separate directory and publish their manifest last.

Software tests cover rank features, passage selection, checkpoint recovery, promotion gates, real ZIP validation, Bash syntax, frozen rclone log copies, retry behavior and failure propagation. GPU installation, Linux process-group shutdown and H100 speed/quality measurements must still be verified on the rental host; local Windows tests cannot establish them.
