@echo off
REM 在 new 目录运行：创建 conda 环境并安装全部模型依赖
REM 若 GRU/LightGCN 损坏，请再运行 repair_deep_models.bat

set ENV_NAME=taxi-models
set PYTHON_VER=3.11

echo [1/4] 创建 conda 环境 %ENV_NAME% (Python %PYTHON_VER%) ...
call conda create -n %ENV_NAME% python=%PYTHON_VER% -y
if errorlevel 1 exit /b 1

echo [2/4] 安装树模型依赖 ...
call conda activate %ENV_NAME%
pip install -r "%~dp0requirements-models.txt"
if errorlevel 1 exit /b 1

echo [3/4] 安装 GRU + LightGCN 依赖 ...
call "%~dp0repair_deep_models.bat"
if errorlevel 1 exit /b 1

echo [4/4] 树模型验证 ...
python -c "import lightgbm, catboost, xgboost; print('tree models OK')"

echo.
echo 使用方式:
echo   conda activate %ENV_NAME%
echo   cd /d %~dp0models
echo   python GRU.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all
echo   python LightGCN.py --inputs ..\may14\xgb_dense_features.csv ..\jun14\xgb_dense_features.csv --feature-mode raw_all

pause
