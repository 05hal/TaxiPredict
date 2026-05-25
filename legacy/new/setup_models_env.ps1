# PowerShell: run from new/
#   Set-ExecutionPolicy -Scope Process Bypass -Force; .\setup_models_env.ps1
#
# If GRU/LightGCN fail with import errors, run: .\repair_deep_models.ps1

$ErrorActionPreference = "Stop"
$EnvName = "taxi-models"
$PythonVer = "3.11"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "[1/4] Create conda env $EnvName (Python $PythonVer) ..."
conda create -n $EnvName python=$PythonVer -y

Write-Host "[2/4] Install tree models (requirements-models.txt) ..."
conda run -n $EnvName pip install -r (Join-Path $Root "requirements-models.txt")

Write-Host "[3/4] Install GRU + LightGCN deps ..."
& (Join-Path $Root "repair_deep_models.ps1")

Write-Host "[4/4] Verify tree models ..."
conda run -n $EnvName python -c "import lightgbm, catboost, xgboost; print('tree models OK')"

Write-Host ""
Write-Host "Done. conda activate $EnvName"
