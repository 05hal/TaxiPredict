# Fix broken TensorFlow (GRU) and PyTorch (LightGCN) in taxi-models
# Run from new/:  Set-ExecutionPolicy -Scope Process Bypass -Force; .\repair_deep_models.ps1

$ErrorActionPreference = "Stop"
$EnvName = "taxi-models"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$TorchCpuIndex = "https://download.pytorch.org/whl/cpu"

function Invoke-CondaPip {
    param([string[]]$PipArgs)
    conda run -n $EnvName pip @PipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip failed: $($PipArgs -join ' ')" }
}

Write-Host "=== Repair $EnvName : GRU + LightGCN ===" -ForegroundColor Cyan

Write-Host "`n[1/4] Uninstall broken packages ..."
$removePkgs = @(
    "torch", "torchvision", "torchaudio",
    "tensorflow", "tensorflow-intel", "keras",
    "tf-keras", "tensorboard"
)
foreach ($pkg in $removePkgs) {
    try {
        Invoke-CondaPip @("uninstall", "-y", $pkg) | Out-Null
    } catch {
        Write-Host "  skip (not installed): $pkg"
    }
}

Write-Host "`n[2/4] Install TensorFlow 2.16.2 (GRU) ..."
Invoke-CondaPip @(
    "install", "--no-cache-dir", "--force-reinstall",
    "-r", (Join-Path $Root "requirements-models-gru.txt")
)

Write-Host "`n[3/4] Install PyTorch 2.4.1 CPU (LightGCN) ..."
Invoke-CondaPip @(
    "install", "--no-cache-dir", "--force-reinstall",
    "torch==2.4.1",
    "--index-url", $TorchCpuIndex
)

Write-Host "`n[4/4] Verify imports ..."
conda run -n $EnvName python (Join-Path $Root "verify_deep_models.py")
if ($LASTEXITCODE -ne 0) {
    throw "Verification failed. Check disk space and network, then retry."
}

Write-Host "`nDone. Example commands:" -ForegroundColor Green
Write-Host "  conda activate $EnvName"
Write-Host "  cd models"
Write-Host "  python GRU.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all"
Write-Host "  python LightGCN.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all"
