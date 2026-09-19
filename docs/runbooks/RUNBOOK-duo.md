# Joint document comparator

This experiment changes the ranking architecture. The existing V9 ensemble proposes sixteen documents; a separately fine-tuned Qwen3-Reranker-0.6B reads the question and **two competing documents together**. It learns from positive-versus-hard-negative comparisons, scores both document orders, and resolves the resulting comparison graph with a regularised Bradley-Terry model. A labelled-question retrieval channel can add up to three confident candidate documents.

This follows the general pointwise-then-pairwise design investigated in [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424). The new comparator, training setup and tournament described here are this project's implementation. That paper's results do not predict gains on this legal dataset. The comparator is derived from the existing [Qwen3-Reranker-0.6B checkpoint](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), preserving the original pointwise partner separately.

## What is measured so far

The CPU prototype in `legalir2/impact.py` replaced linear fusion with query-conditioned residual trees, a Recall@5-weighted pairwise objective, both passage views and labelled-question retrieval. Its three grouped training folds improved from 0.954705 to 0.955622, but validation stayed at **0.946032**. It is not a promoted model and needs no paid public scoring. The results are retained in `analysis/impact/run-01/`.

The joint comparator is the subsequent, larger experiment. On the existing validation cache, its proposed shortlist has **0.975714 oracle recall**, versus V9's 0.946032 final recall. The 0.029683 gap is the maximum available improvement within that shortlist under perfect ranking, **not a predicted gain**. It averages 16.01 candidates and 240.31 ordered pair evaluations per question. Its trained quality and H100 runtime have not yet been measured.

## Training and evaluation

- Reuse the completed H100 run's pointwise scores. No repeat of the old calibration GPU pass is needed.
- Split its 1,998 calibration questions by normalised question groups: **1,569 training, 429 development**. Exact duplicate text stays in one split. Validation questions and their duplicate texts are excluded from question-memory labels, as are the development questions.
- Mine up to six hard negative documents from the pointwise ranking plus two from question retrieval for each gold document. Never label another gold document as negative. Train both A/B presentations with equal total weight per query.
- Fine-tune a rank-16 LoRA adapter on the existing Qwen partner for two epochs, selecting the checkpoint by development pair loss. The original checkpoint remains intact. Merge the chosen adapter into a separate comparator checkpoint.
- Give the prompt at most 2,048 tokens: a reserved query budget and equal document budgets. Each side includes up to three diverse articles, allocating 60% of its evidence budget to the best article and redistributing unused tokens. Both presentation orders see the same evidence.
- Fit the tournament's blend strength on development queries only. If the best strength is zero, stop before validation and public inference.
- Evaluate the chosen comparator on all 1,050 validation questions. Generate a new public ZIP only when the paired 95% lower bound exceeds zero. This remains exploratory because the validation split has already been reused across experiments; a public score increase is not guaranteed.

At deployment the original three neural models plus the separate comparator total approximately **2.327B parameters**, below the existing 4B budget. The script audits the loaded comparator's actual parameter count against the original model audit.

## Run on the H100

Extract `duo-overlay.zip` into the **existing** `/workspace/DSC-2026-h100` workspace. The overlay adds files and preserves the previous experiment's source files, indexes, models and original integrity manifest. It contains no credentials or model weights. The original H100 workspace and configured `gdrive:` remote are prerequisites.

Upload the overlay from Windows PowerShell using the existing local rclone config:

```powershell
Set-Location D:\DSC-2026
.\tools\bin\rclone.exe --config .\.secrets\rclone.conf copyto .\analysis\impact\duo-overlay.zip gdrive:DSC-2026-results/code/duo-overlay.zip
```

Then download and launch it on the H100:

```bash
cd /workspace/DSC-2026-h100
rclone copyto gdrive:DSC-2026-results/code/duo-overlay.zip duo-overlay.zip
unzip -o duo-overlay.zip
sha256sum -c --quiet duo-files.sha256
tmux new -s duo
SOURCE_RUN_ID=20260918T013034Z bash scripts/gpu/gpu-duo-h100.sh
```

Default stages are **prepare → train → calibrate → validate → submit**. Required CPU dependencies are installed into `.venv-h100` while Torch, Transformers and NumPy remain pinned. Preflight refuses changes to the original model/index/scoring files and key package versions. SDPA is the default backend; `DUO_ATTN=flash_attention_2` is optional and must remain unchanged when resuming. No FlashAttention speedup is claimed for this new workload.

Run `bash scripts/gpu/gpu-duo-h100.sh --help` for options. Set `RCLONE_CONFIG=/path/to/rclone.conf` if the config is outside rclone's default location. The runner checks CUDA/BF16 support before dependency installation. Credentials are not part of the uploaded artifacts.

The first launch writes `outputs/.duo-active`. Run the same command to resume. Training checkpoints contain the adapter, optimizer, scheduler, epoch/batch cursor, and Torch CPU/CUDA RNG state every 100 optimizer steps. An interruption can lose up to that interval. Scoring saves ten-question shards and reuses completed shards. Logs upload automatically and model/checkpoint transfers have a twenty-minute deadline. A failed final upload returns nonzero and retains local artifacts.

```bash
bash scripts/gpu/gpu-duo-h100.sh prepare
bash scripts/gpu/gpu-duo-h100.sh train
bash scripts/gpu/gpu-duo-h100.sh calibrate
bash scripts/gpu/gpu-duo-h100.sh validate
bash scripts/gpu/gpu-duo-h100.sh submit
```

Individual stages require their earlier results. `all` is the normal entry point. `RUN_ID` starts a separate experiment; `EPOCHS`, `TRAIN_BATCH`, `ACCUMULATE`, `INFER_BATCH` and `PAIR_LENGTH` are configurable. Changing the training configuration requires a new run ID. Do not run the original launcher concurrently; both share its workspace/GPU lock.

Training prints a measured seconds-per-optimizer-step rate and an ETA after twenty steps. That ETA covers remaining training updates; development scoring, validation, public inference and uploads add time. There is no measured end-to-end H100 estimate yet.

Results are in `outputs/duo/<RUN_ID>/` and `gdrive:DSC-2026-results/runs/duo/<RUN_ID>/artifacts/`. Important files: `calibration.json`, `selection.json`, `audit.json`, `model/duo.json`, `duo.diagnostics.json`, and **`submission-duo.zip`**. Live status is at the run's `logs/STATUS.txt`.

If the comparator fails its gates, **no new submission is generated**. The launcher explicitly reports this and never repackages V9 under a new submission name.

An explicit user decision can override an inconclusive positive validation result with `bash scripts/gpu/gpu-duo-h100.sh submit-approved`. This uses the existing development-selected strength and trained checkpoint, skips retraining and validation, and retains all input/model integrity checks. It preserves `selection.json` and records the decision and original statistics in `submission-approval.json`. Public scoring remains resumable and the validated ZIP is uploaded normally. The default `all` and `submit` stages retain automatic statistical gating. The approved stage generates the competition ZIP; it does not upload to the competition website.

## Restored partner provenance

The published Qwen partner may be downloaded again when the model cloud backup is absent. Its `provenance.json` includes download timing, so that receipt can change while all inference files remain identical. Preparation accepts a changed partner receipt only after every source-run model, tokenizer, config and other input file passes its original SHA-256 check, and the receipt declares the expected Qwen repository and exact source file hashes. Other provenance files remain strict. A changed model file still stops the run; source manifests are never rewritten to bless changed weights.

The earlier receipt-hash failure occurs before comparator training or its input manifest is created. After installing the repaired overlay, rerun with the same source/run ID and attention setting. A comparator run already prepared under older code needs a new run ID because its code signature is immutable.

## Local verification limits

Tests exercise prompt budgets, document-order symmetry, a real tiny Qwen forward/backward pass, tournament cycles and permutation invariance, leakage exclusions, paired-loss weighting, checkpoint validation and shell failure handling. Full LoRA training, optimizer resume on CUDA, merged-checkpoint inference and end-to-end quality must still be verified on the rental GPU. The comparator is an implemented experiment, not a demonstrated score improvement.
