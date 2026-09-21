# DSC 2026 — start here

Current local developer handoff: [development/README.md](development/README.md).
It records the 0.9510 best private result and the subsequent 256/512 regressions.
The snapshot paths below describe the original 19 September layout, not the
latest experiment recommendation.

- `source/`: current organized source tree, scripts, config, docs and tests.
- `source/docs/runbooks/RUNBOOK-private.md`: original private GPU instructions.
- `archive/flat-workspace-20260919/`: the previous flat workspace snapshot, preserved.

The sibling `DSC-2026-results` contains:

- `submissions/private/20260919-v1/`: duo **0.9469**, V9 **0.9457**.
- `releases/private/20260919-layout-v2/`: original layout-v2 inference installer.
- `runs/private/`: private run caches and logs.
- `assets/models/`: trained/base model backups.
- `docs/PROJECT-LAYOUT.md`: full directory map.

Old root-level paths moved during the layout-v2 migration. Use the updated release.
AIC and NCKH are separate projects and must not be modified.
