param(
    [string]$Config = "configs/base.yaml",
    [string]$RunConfig = "configs/run.yaml"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Project .venv was not found." }
Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

Write-Host "Training shared GCN from existing selected attacks. No attacks will be generated."
& $pythonExe -u -m gnn_attack_detector.cli rebuild-detector --config $Config --run-config $RunConfig
if ($LASTEXITCODE -ne 0) { throw "Detector rebuild failed. Review the error above; existing attacks are unchanged." }

$archiveScript = Join-Path $PSScriptRoot "archive_detector_result.ps1"
& $archiveScript -ProjectRoot $projectRoot
if ($LASTEXITCODE -ne 0) { throw "Detector result archiving failed. The detector was trained, but its result was not archived." }

Write-Host "Detector trained and active model updated automatically."
Write-Host "Existing per-profile reports are retained; a selected profile is updated only when it is trained again."
