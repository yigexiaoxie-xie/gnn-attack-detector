$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "Creating the project virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.10 -m venv (Join-Path $projectRoot ".venv")
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $projectRoot ".venv")
    } else {
        throw "Python 3.10 was not found. Install Python or create .venv in PyCharm."
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $pythonExe)) {
        throw "Virtual environment creation failed."
    }
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:NO_PROXY = "*"

Write-Host "Installing the pinned CPU toolkit dependencies..."
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --timeout 120 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Installing PyG sparse wheels required by PRBCD and GreatX..."
& $pythonExe -m pip --isolated install --no-index --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html torch-sparse==0.6.18 torch-cluster==1.6.3
if ($LASTEXITCODE -ne 0) { throw "PyG sparse wheel installation failed." }

Write-Host "Installing DeepRobust without its obsolete transitive dependency pins..."
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --timeout 120 --no-deps deeprobust==0.2.9
if ($LASTEXITCODE -ne 0) { throw "DeepRobust installation failed." }

Write-Host "Installing the pinned GreatX adapter backend without changing PyTorch..."
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --timeout 120 --no-deps greatx==0.0.1b0
if ($LASTEXITCODE -ne 0) { throw "GreatX installation failed." }

Write-Host "Installing this project without rebuilding dependencies..."
& $pythonExe -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

& $pythonExe -c "import torch, torch_geometric, torch_cluster, torch_sparse; from deeprobust.graph.defense import GCNJaccard, SimPGCN; from deeprobust.graph.global_attack import DICE, PGDAttack, PRBCD, Random; from deeprobust.graph.targeted_attack import FGA, IGAttack, Nettack, RND, SGAttack; from greatx.attack.targeted import GFAttack; from greatx.attack.untargeted import FGAttack; from gnn_attack_detector.attacks.registry import SUPPORTED_ATTACKS; from gnn_attack_detector.defenses.registry import SUPPORTED_DEFENSES; print(f'Environment ready: {len(SUPPORTED_ATTACKS)} attacks, {len(SUPPORTED_DEFENSES)} defenses, DeepRobust and GreatX')"
if ($LASTEXITCODE -ne 0) { throw "Import verification failed." }
