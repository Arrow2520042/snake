$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
$VenvCfg = Join-Path $VenvDir "pyvenv.cfg"

$NeedsRecreate = $false
if (Test-Path $VenvCfg) {
    $cfgText = Get-Content -Path $VenvCfg -Raw
    if (-not $cfgText.Contains($VenvDir)) {
        $NeedsRecreate = $true
    }
}

if ($NeedsRecreate -and (Test-Path $VenvDir)) {
    Write-Host "[0/3] Recreating virtual environment (.venv) due to stale path in pyvenv.cfg"
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[0/3] Creating virtual environment (.venv)"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvDir
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    }
    else {
        throw "Python launcher not found. Install Python 3 and retry."
    }
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found at $VenvPython"
}

Write-Host "[1/3] Installing runtime/build dependencies"
& $VenvPython -m pip install --upgrade pip wheel
& $VenvPython -m pip install --upgrade 'setuptools<81'
& $VenvPython -m pip install --upgrade numpy cython numba pygame torch

Write-Host "[2/3] Building per_cython backend"
& $VenvPython (Join-Path $RootDir "setup_cython_per.py") build_ext --inplace

Write-Host "[3/3] Verifying accelerators"
& $VenvPython (Join-Path $RootDir "verify_runtime.py")

Write-Host "Done. Windows accelerator setup is complete."
