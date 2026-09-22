# Repair the retraining release's missing imports

The September 21 retraining release omitted `legalir2/train_ce.py`, imported
inside `duo.train`, and its dependencies `train_bi.py` and `mine.py`. Preparation
passed because it did not enter that function. Training failed before loading
the model or performing an optimizer update. This is a release packaging defect.

The additive repair is published under
`gdrive:DSC-2026-results/releases/tools/20260921-retrain-training-deps-v1`.
It adds the three original modules and records their checksums. It verifies the
original manifest and Python files, refuses conflicting files and active jobs,
and holds the bootstrap/workspace locks while installing. It preserves the
original ZIP, manifest, prepared inputs, model weights, cached comparisons and
checkpoints. Do not edit `duo.py` or regenerate inputs to bypass this error.

On the existing GPU, after the failed job finishes its final backup:

```bash
(
set -euo pipefail
mkdir -p /workspace/dsc-training-deps-fix /workspace/dsc-retrain-launcher-v2
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-training-deps-v1 /workspace/dsc-training-deps-fix --progress
cd /workspace/dsc-training-deps-fix
sha256sum -c files.sha256
python3 repair_retrain_dependencies.py --root /workspace/DSC-2026-fulltext-retrain-4090-v1
rclone copy gdrive:DSC-2026-results/releases/tools/20260921-retrain-launcher-v2 /workspace/dsc-retrain-launcher-v2 --progress
cd /workspace/dsc-retrain-launcher-v2
sha256sum -c files.sha256
python3 launch_retrain.py
)
```

The installer checks real training and PEFT imports inside `.venv-h100`. The
launcher preserves existing assets and resumes the same run. If prompted, enter
the Vast API key at the hidden terminal prompt. Confirm `AUTO-STOP ARMED` or
`AUTO-STOP ALREADY ARMED` and subsequent training progress. A failure keeps the
instance running; successful completion plus verified backup permits STOP.

The CPU regression test extracts Python files from the actual immutable ZIP
into an isolated directory, reproduces `ModuleNotFoundError`, applies the repair,
then calls the packaged `duo.train` again and reaches its expected BF16 CUDA
guard. Only the unavailable external PEFT exports are stubbed, and no PEFT
function executes. The test also verifies idempotence, original-file preservation
and refusal to overwrite a differing dependency. This is not a GPU training test.

Future packages now traverse relative imports, including function-local imports,
to include the full transitive dependency closure. Already published releases
remain immutable.
