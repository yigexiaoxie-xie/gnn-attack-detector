$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$downloadScript = Join-Path $PSScriptRoot "download_official_metattack.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project virtual environment was not found: $pythonExe"
}

Set-Location -LiteralPath $projectRoot
& $pythonExe $downloadScript
if ($LASTEXITCODE -ne 0) {
    throw "Official Metattack graph preparation failed. Review the error above."
}
