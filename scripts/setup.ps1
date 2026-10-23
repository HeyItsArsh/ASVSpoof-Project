$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath ".venv/Scripts/python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 or 3.12 is required" }
}
$requirements = if (Test-Path -LiteralPath "requirements-lock.txt") { "requirements-lock.txt" } else { "requirements.txt" }
& ./.venv/Scripts/python.exe -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
Write-Output "Setup complete. Use ./.venv/Scripts/python.exe for all Python commands."
