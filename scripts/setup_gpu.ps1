$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath ".venv-gpu/Scripts/python.exe")) {
    python -m venv .venv-gpu
    if ($LASTEXITCODE -ne 0) { throw "Creating the GPU environment failed" }
}
$gpuPython = (Resolve-Path ".venv-gpu/Scripts/python.exe").Path
if (Test-Path -LiteralPath "requirements-gpu-lock.txt") {
    & $gpuPython -m pip install -r requirements-gpu-lock.txt --timeout 120 --retries 8
} else {
    & $gpuPython -m pip install --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126 --timeout 120 --retries 8
    if ($LASTEXITCODE -ne 0) { throw "Installing CUDA PyTorch failed" }
    & $gpuPython -m pip install -r requirements-lock.txt --timeout 120 --retries 8
}
if ($LASTEXITCODE -ne 0) { throw "Installing GPU environment dependencies failed" }
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "GPU dependency check failed" }
& $gpuPython scripts/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw "GPU verification failed" }
Write-Output "GPU setup verified. Resume using ./.venv-gpu/Scripts/python.exe scripts/run_all.py --resume"
