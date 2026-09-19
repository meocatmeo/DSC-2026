# DSC 2026 — LegalIR

**Private test: 2,080 questions. Best reported score: duo 0.9469; V9 0.9457.**

- [GPU setup and inference](docs/runbooks/RUNBOOK-private.md)
- [Submission files and scores](submissions/private/20260919-v1/README.md)
- [Directory map and migration details](docs/PROJECT-LAYOUT.md)
- [Private task state](docs/state/PRIVATE-TEST.md)
- [Historical system summary](docs/state/SUMMARY.md)

## Local layout

| Folder | Purpose |
|---|---|
| `scripts/gpu/` | GPU launchers and runtime helpers |
| `scripts/package/` | Python release builders |
| `scripts/storage/` | DSC-only cloud migration and publishing |
| `config/` | Dependency pins |
| `data/` | Train, public and private question files; local corpus |
| `docs/runbooks/`, `docs/state/`, `docs/history/`, `docs/reference/` | Current instructions, state, earlier work and competition documents |
| `legalir2/`, `tests2/` | Current Python pipeline and tests |
| `legalir/`, `tests/` | Original Python package and tests |
| `tools/scoring/` | Official scorer and its metadata |
| `submissions/`, `releases/`, `reports/` | Generated deliverables, installers and reports (ignored by Git) |
| `analysis/` | Research/build working files (ignored) |
| `outputs/`, `artifacts*/`, `models*/` | Existing runtime state (ignored; byte-sensitive paths kept) |
| `archive/` | Previous upload workspaces and miscellaneous old files (ignored) |
| `.secrets/` | Local rclone configuration (ignored; never published) |
| `tools/bin/` | Local rclone executable (ignored; never published) |

Python source remains tracked; all shell scripts are ignored by Git at your request. Ignore rules also cover secrets, binaries, environments,
model weights, caches and generated outputs. Shell scripts are preserved locally and in the cloud source/release packages.

## Current commands

```bash
bash scripts/gpu/gpu-private-setup.sh all  # fresh H100
bash scripts/gpu/gpu-private-h100.sh all   # existing compatible environment
python scripts/package/package-private.py
```

Run commands from the repository root. Launchers also locate the root when called
using an absolute path from another directory. The private package builder checks
all signed V9 and duo inference module hashes before it creates a release.

## Rclone layout

`gdrive:DSC-2026`: `source/` contains the organized code; `archive/` retains the previous
flat workspace snapshot byte-for-byte.

`gdrive:DSC-2026-results`: `assets/`, `runs/`, `releases/`, `submissions/`, `reports/`,
`docs/` and `archive/`. Use `runs/private/` for live private logs and caches.

**The old root-level cloud paths have moved. Download the layout-v2 release before
resuming an older GPU workspace.** See the current runbook for exact commands.
AIC and NCKH are separate projects and are excluded from all storage operations.
