[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RootDir ".venv/Scripts/python.exe"
$VenvCfg = Join-Path $RootDir ".venv/pyvenv.cfg"
$ExpectedVenv = Join-Path $RootDir ".venv"
$NeedsBootstrap = $false

if (Test-Path $VenvCfg) {
    $cfgText = Get-Content -Path $VenvCfg -Raw
    if (-not $cfgText.Contains($ExpectedVenv)) {
        $NeedsBootstrap = $true
    }
}

if ((-not (Test-Path $VenvPython)) -or $NeedsBootstrap) {
    Write-Host "Virtual environment not found. Running bootstrap_windows.ps1..."
    & (Join-Path $RootDir "bootstrap_windows.ps1")
    $VenvPython = Join-Path $RootDir ".venv/Scripts/python.exe"
}

& $VenvPython (Join-Path $RootDir "train.py") @TrainArgs
exit $LASTEXITCODE
