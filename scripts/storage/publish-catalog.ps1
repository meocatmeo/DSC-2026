param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$rclone = Join-Path $root 'tools/bin/rclone.exe'
$config = Join-Path $root '.secrets/rclone.conf'
$base = 'gdrive:DSC-2026-results'

# Explicit small deliverables only. Never traverse or upload the workspace root.
$jobs = @(
    @{source='submissions/private/20260919-v1'; target="$base/submissions/private/20260919-v1"; directory=$true},
    @{source='releases/private/20260919-qwen-restore'; target="$base/releases/private/20260919-qwen-restore"; directory=$true},
    @{source='releases/private/20260919-layout-v2'; target="$base/releases/private/20260919-layout-v2"; directory=$true},
    @{source='reports/private/20260919-v1'; target="$base/reports/private/20260919-v1"; directory=$true},
    @{source='docs'; target="$base/docs"; directory=$true},
    @{source='reports/organization/CLOUD-README.md'; target="$base/README.md"; directory=$false},
    @{source='docs/DSC-SNAPSHOT-START-HERE.md'; target='gdrive:DSC-2026/START-HERE.md'; directory=$false}
)
foreach ($folder in @('legalir','legalir2','tests','tests2','scripts','config','docs')) {
    $jobs += @{source=$folder;target="gdrive:DSC-2026/source/$folder";directory=$true}
}
foreach ($name in @('private-official.json','public-official.json','train.json')) {
    $jobs += @{source="data/$name";target="gdrive:DSC-2026/source/data/$name";directory=$false}
}
$jobs += @{source='tools/scoring';target='gdrive:DSC-2026/source/tools/scoring';directory=$true}
$jobs += @{source='README.md';target='gdrive:DSC-2026/source/README.md';directory=$false}
$jobs += @{source='.gitignore';target='gdrive:DSC-2026/source/.gitignore';directory=$false}
foreach ($job in $jobs) {
    $source = [IO.Path]::GetFullPath((Join-Path $root $job.source))
    if (-not $source.StartsWith($root+[IO.Path]::DirectorySeparatorChar) -or -not (Test-Path -LiteralPath $source)) {
        throw "Invalid local source: $source"
    }
    if (-not ($job.target.StartsWith('gdrive:DSC-2026-results/') -or $job.target.StartsWith('gdrive:DSC-2026/source/') -or $job.target -eq 'gdrive:DSC-2026/START-HERE.md')) {
        throw "Remote outside authorized DSC folders: $($job.target)"
    }
    if ($job.directory) {
        if (Get-ChildItem -LiteralPath $source -Recurse -File | Where-Object { $_.Extension -in '.conf','.exe','.safetensors','.bin' }) {
            throw "Unexpected credential, executable or model in publication directory: $source"
        }
    }
}
foreach ($job in $jobs) {
    $source = Join-Path $root $job.source
    Write-Output "$($job.source) -> $($job.target)"
    if (-not $Apply) { continue }
    $verb = if ($job.directory) { 'copy' } else { 'copyto' }
    $filters = if ($job.directory) { @('--exclude','**/__pycache__/**','--exclude','*.pyc') } else { @() }
    & $rclone --config $config $verb $source $job.target --checksum --retries 2 @filters
    if ($LASTEXITCODE -ne 0) { throw "Publication failed: $($job.target)" }
    if ($job.directory) {
        & $rclone --config $config check $source $job.target --one-way --exclude '**/__pycache__/**' --exclude '*.pyc'
        if ($LASTEXITCODE -ne 0) { throw "Remote verification failed: $($job.target)" }
    } else {
        $remoteHash = & $rclone --config $config md5sum $job.target
        if ($LASTEXITCODE -ne 0) { throw "Cannot verify remote file: $($job.target)" }
        $localHash = (Get-FileHash -LiteralPath $source -Algorithm MD5).Hash.ToLower()
        if (-not ($remoteHash -match "^$localHash\s")) { throw "Remote checksum mismatch: $($job.target)" }
    }
}
if ($Apply) { Write-Output 'Catalog published and verified. Only DSC destinations were modified.' }
else { Write-Output 'Preview only. Use -Apply to publish these explicit paths.' }
