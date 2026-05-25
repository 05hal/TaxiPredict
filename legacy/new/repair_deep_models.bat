@echo off
REM 修复 taxi-models 中损坏的 TensorFlow / PyTorch
REM 在 new 目录双击或：repair_deep_models.bat

set ENV_NAME=taxi-models
set TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

echo === 修复 %ENV_NAME%：GRU + LightGCN ===
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo 请先创建环境: setup_models_env.bat
    pause
    exit /b 1
)

echo.
echo [1/4] 卸载损坏包 ...
pip uninstall -y torch torchvision torchaudio tensorflow tensorflow-intel keras tf-keras tensorboard 2>nul

echo.
echo [2/4] 安装 TensorFlow（GRU）...
pip install --no-cache-dir --force-reinstall -r "%~dp0requirements-models-gru.txt"
if errorlevel 1 goto :fail

echo.
echo [3/4] 安装 PyTorch CPU（LightGCN）...
pip install --no-cache-dir --force-reinstall torch==2.4.1 --index-url %TORCH_CPU_INDEX%
if errorlevel 1 goto :fail

echo.
echo [4/4] 验证 ...
python "%~dp0verify_deep_models.py"
if errorlevel 1 goto :fail

echo.
echo 完成。示例:
echo   cd /d %~dp0models
echo   python GRU.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all
echo   python LightGCN.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all
pause
exit /b 0

:fail
echo.
echo 修复失败。请确认磁盘空间充足后重试 repair_deep_models.bat
pause
exit /b 1
