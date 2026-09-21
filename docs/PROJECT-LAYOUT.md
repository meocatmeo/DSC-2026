# Project layout — version 2

Current development handoff (21 September 2026):
[docs/development/README.md](development/README.md). The map below records the
19 September organization; later experiments have additional immutable releases.

The 19 September 2026 migration physically moved loose scripts, Python entry points,
data, credentials and documents into folders, and reorganized both DSC Drive roots.

## Local entry points

- `scripts/gpu/gpu-private-setup.sh`: fresh H100 environment, restore and inference.
- `scripts/gpu/gpu-private-h100.sh`: restore, preflight, V9, duo and submit.
- `scripts/gpu/gpu-duo-h100.sh`: historical comparator training/public submission.
- `scripts/gpu/gpu-ranking-h100.sh`: historical ranking experiment.
- `scripts/package/package-private.py`: immutable-model private release builder.
- `scripts/storage/publish-catalog.ps1`: preview or publish the explicit DSC allowlist.
- `scripts/storage/migrate-cloud-layout.ps1`: recorded one-time cloud migration.

Runtime helpers and historical GPU experiments share `scripts/gpu/`; filenames retain
 their original purpose/version. Documentation is grouped under `docs/runbooks/`,
`docs/state/`, `docs/history/` and `docs/reference/`. Dependencies live in `config/`.
Question files live in `data/`. The official scorer and metadata stay together in
`tools/scoring/`. Source packages retain normal import paths (`legalir`, `legalir2`).

`.secrets/rclone.conf` and `tools/bin/rclone.exe` are ignored and never published.
Generated artifacts, models, environments, caches, releases and submissions are also
ignored. All *.sh scripts are also ignored by explicit user request; Python and PowerShell source remain tracked. Shell scripts are retained in local folders and cloud source/release packages.

## Cloud directory map

```text
DSC-2026/
  START-HERE.md
  source/                     organized code, scripts, config, data, docs and tests
  archive/flat-workspace-20260919/  previous flat snapshot, preserved

DSC-2026-results/
  README.md
  assets/
    models/                   all shared model backups
    indexes/dense/            original dense index
    indexes/dense-tuned/      tuned dense index
    training/groups/          training groups
    training/cache/           fold caches
  runs/
    private/                  scored private run, shards and logs
    setup/private/            environment setup attempts
    duo/                      trained comparator checkpoints and signatures
    ranking-h100/             exact source indexes and ranking experiment
    v11/                      earlier V11 runs
  releases/
    private/20260919-layout-v2/   original layout-v2 inference package
    private/20260919-qwen-restore/ earlier package (old cloud paths)
    ranking-h100/             historical ready bundle
    legacy/code/              earlier comparator patches and overlays
  submissions/
    private/20260919-v1/       duo 0.9469 and V9 0.9457
    public/manual/            older manual public submissions
  reports/private/20260919-v1/
  docs/
  archive/legacy/
    logs/
    outputs/
```

Only `DSC-2026` and `DSC-2026-results` are authorized. **AIC and NCKH are untouched.**

## Compatibility and resuming

Cloud paths changed; old downloaded launchers must be refreshed with the new release.
The local GPU model/index/output directories retain their original paths because the
saved pipeline signatures rely on their contents and conventions. Model weights,
indexes, scored predictions and signed V9/duo inference modules are unchanged.
The new launcher passes `data/private-official.json` explicitly and uses the new cloud
paths. Existing completed shards remain reusable when the saved inputs match.
Historical documents/releases preserve the paths used at their creation date.
Use current runbooks rather than old copied commands.

## Audit and rollback

`reports/organization/` contains complete before/after recursive Drive inventories,
local move records, the cloud move plan/completion log, and hash verification results.
To reverse a completed move, inspect that record and move its destination back to the
original source only when the original path is absent. Do not merge conflicting files.
Nothing in AIC or NCKH is included in a migration or rollback plan.
