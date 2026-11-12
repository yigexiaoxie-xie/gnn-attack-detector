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
}

Set-Location -LiteralPath $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
$env:NO_PROXY = "*"

Write-Host "Installing pinned project dependencies..."
& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --timeout 120 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

Write-Host "Replacing CPU PyTorch with CUDA 12.1 PyTorch..."
& $pythonExe -m pip --isolated install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

Write-Host "Installing CUDA PyG sparse wheels..."
& $pythonExe -m pip --isolated install --force-reinstall --no-deps --no-index --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html torch-sparse==0.6.18 torch-cluster==1.6.3
if ($LASTEXITCODE -ne 0) { throw "CUDA PyG sparse wheel installation failed." }

& $pythonExe -m pip --isolated install --index-url https://pypi.org/simple --timeout 120 --no-deps deeprobust==0.2.9 greatx==0.0.1b0
if ($LASTEXITCODE -ne 0) { throw "Attack backend installation failed." }

& $pythonExe -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

& $pythonExe -c "import torch, torch_geometric, torch_cluster, torch_sparse; assert torch.cuda.is_available(), 'CUDA PyTorch cannot see the NVIDIA GPU'; print('Environment ready:', torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA verification failed." }
