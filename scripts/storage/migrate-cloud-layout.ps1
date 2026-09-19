param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$rclone = Join-Path $root 'tools/bin/rclone.exe'
$config = Join-Path $root '.secrets/rclone.conf'
$directory = Join-Path $root 'reports/organization'
$results = Get-Content "$directory/cloud-results-before.json" -Raw | ConvertFrom-Json
$snapshot = Get-Content "$directory/cloud-source-before.json" -Raw | ConvertFrom-Json
$map = [ordered]@{
    'models'='assets/models'; 'groups'='assets/training/groups'; 'cache'='assets/training/cache'
    'dense'='assets/indexes/dense'; 'dense-tuned'='assets/indexes/dense-tuned'
    'train.stats.json'='assets/training/train.stats.json'
    'private'='runs/private'; 'private-setup'='runs/setup/private'; 'duo'='runs/duo'
    'ranking-h100'='runs/ranking-h100'; 'v11'='runs/v11'
    'outputs'='archive/legacy/outputs'; 'logs'='archive/legacy/logs'
    'manual'='submissions/public/manual'; 'code'='releases/legacy/code'
    'ranking-h100-release'='releases/ranking-h100'
}
$jobs = @()
foreach ($item in $map.GetEnumerator()) {
    if (-not ($results | Where-Object Path -eq $item.Key)) { throw "Missing inventoried source: $($item.Key)" }
    $jobs += [ordered]@{remote='gdrive:DSC-2026-results';source=$item.Key;destination=$item.Value}
}
foreach ($item in $snapshot | Where-Object { $_.Path -notmatch '/' }) {
    $jobs += [ordered]@{remote='gdrive:DSC-2026';source=$item.Path;destination='archive/flat-workspace-20260919/'+$item.Path}
}
foreach ($job in $jobs) {
    if ($job.remote -notin @('gdrive:DSC-2026','gdrive:DSC-2026-results')) { throw 'Unauthorized remote root' }
    if ($job.source -match '(^/|\.\.|:)' -or $job.destination -match '(^/|\.\.|:)') { throw 'Invalid move path' }
    $inventory = if ($job.remote -eq 'gdrive:DSC-2026') { $snapshot } else { $results }
    if ($inventory | Where-Object { $_.Path -eq $job.destination -or $_.Path.StartsWith($job.destination+'/') }) {
        throw "Destination already occupied in inventory: $($job.destination)"
    }
}
$jobs | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$directory/cloud-layout-v2-moves.json"
foreach ($job in $jobs) {
    $source = "$($job.remote)/$($job.source)"
    $target = "$($job.remote)/$($job.destination)"
    Write-Output "$source -> $target"
    if (-not $Apply) { continue }
    # Same-Drive moves use server-side directory moves when supported.
    # Never merge into a pre-existing destination: unexpected state needs inspection.
    $existing = & $rclone --config $config lsjson $target --stat 2>$null
    if ($LASTEXITCODE -eq 0) { throw "Destination already exists: $target" }
    if ($LASTEXITCODE -notin @(3,4)) { throw "Cannot verify destination absence: $target (exit $LASTEXITCODE)" }
    & $rclone --config $config moveto $source $target --retries 2 --stats 30s
    if ($LASTEXITCODE -ne 0) { throw "Move failed: $source" }
    $job | ConvertTo-Json -Compress | Add-Content -Encoding utf8 "$directory/cloud-layout-v2-completed.jsonl"
}
if ($Apply) { Write-Output 'Cloud moves finished; verify against the before inventories.' }
else { Write-Output 'Move plan only; no remote modifications. Use -Apply for the listed DSC-only moves.' }
