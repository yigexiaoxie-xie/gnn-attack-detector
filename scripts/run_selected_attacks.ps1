$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project .venv was not found. Run scripts\setup_windows_cpu.bat first."
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:MPLCONFIGDIR = Join-Path $projectRoot "artifacts\cache\matplotlib"
$env:NUMBA_CACHE_DIR = Join-Path $projectRoot "artifacts\cache\numba"
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:NO_PROXY = "*"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:NUMBA_CACHE_DIR | Out-Null

Write-Host "Selected run preview:"
& $pythonExe -m gnn_attack_detector.cli show-selection --config configs/base.yaml --run-config configs/run.yaml
if ($LASTEXITCODE -ne 0) { throw "Selection preview failed." }

Write-Host "Starting the resumable selected attack workflow..."
& $pythonExe -m gnn_attack_detector.cli run-selected --config configs/base.yaml --run-config configs/run.yaml
if ($LASTEXITCODE -ne 0) { throw "Selected attack workflow failed. Review the error summary above." }

Write-Host "Selected attack workflow completed."
Write-Host "Measured detector results:"
& $pythonExe show_results.py
if ($LASTEXITCODE -ne 0) { Write-Warning "The workflow finished, but no current test evaluation could be displayed." }
