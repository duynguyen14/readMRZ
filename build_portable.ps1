$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pyInstallerCheck = python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    python -m pip install pyinstaller
}

python -m PyInstaller `
    --clean `
    --noconfirm `
    --name readmrz `
    --collect-all rapidocr_onnxruntime `
    --hidden-import rapidocr_onnxruntime `
    readmrz.py

Write-Host "Portable build created at: $projectRoot\dist\readmrz\readmrz.exe"
