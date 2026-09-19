# Script entry points

Run from the repository root; paths below are relative to it.

All `.sh` files are ignored by Git at the user's request. They remain in the local
`scripts/gpu/` directory and in rclone source/release packages. After a Git-only clone,
download `gdrive:DSC-2026/source/scripts/gpu` into `scripts/gpu/` before using the
commands below. Python and PowerShell source in this directory remains tracked.

## Current private inference

```bash
bash scripts/gpu/gpu-private-setup.sh all
bash scripts/gpu/gpu-private-h100.sh all
```

The first command installs a fresh H100 environment; the second reuses an existing
compatible environment. Both restore saved models, generate V9 then duo predictions,
execute the submit stage and back up through the reorganized DSC rclone paths.
Do not run both concurrently. See `docs/runbooks/RUNBOOK-private.md`.

## Release builders

```bash
python scripts/package/package-private.py
python scripts/package/package-duo.py
python scripts/package/package-ranking-h100.py
```

Build inputs include the saved local experiment signatures and baseline artifacts in
`analysis/`. Private packaging verifies frozen code hashes; it does not change models.

## Storage

```powershell
.\scripts\storage\publish-catalog.ps1          # show exact source/destination pairs
.\scripts\storage\publish-catalog.ps1 -Apply   # publish and verify the allowlist
```

`migrate-cloud-layout.ps1` records the one-time layout-v2 migration. It is not a
routine sync command: the completed move log and original inventories are required.
Both scripts reject paths outside the two DSC cloud roots. AIC and NCKH are excluded.

## Historical GPU experiments

The remaining `gpu-*` scripts retain descriptive experiment/version names. Their
local sibling and cloud references have been migrated. They may train or evaluate
historical public workflows; they are not the private submission entry point.
Existing release ZIPs under `releases/legacy/` retain original content and paths.
