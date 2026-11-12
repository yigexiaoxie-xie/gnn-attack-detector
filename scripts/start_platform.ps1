$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project .venv was not found: $pythonExe"
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

$platformPort = $null
foreach ($candidatePort in 8501..8510) {
    $listener = Get-NetTCPConnection -LocalPort $candidatePort -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) {
        $platformPort = $candidatePort
        break
    }
}
if ($null -eq $platformPort) {
    throw "No available platform port was found in the range 8501-8510."
}
$env:GNN_PLATFORM_PORT = [string]$platformPort

Write-Host "Starting the graph robustness toolkit on port $platformPort..."
Write-Host "Server browser: http://localhost:$platformPort"
Write-Host "Another computer: http://SERVER_LAN_IP:$platformPort"
Write-Host "Press Ctrl+C to stop the platform."
& $pythonExe run_platform.py
if ($LASTEXITCODE -ne 0) {
    throw "Platform process exited with an error."
}
