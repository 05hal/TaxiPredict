# TaxiPredict — NYC Uber Taxi Demand Forecasting

基于纽约 Uber 2014 年 5-6 月订单数据，预测各区域逐小时客流量。

## 快速开始

```powershell
:: 1. 环境
conda create -n taxi_predict python=3.10
conda activate taxi_predict
pip install -r requirements/base.txt
pip install -r requirements/tree.txt

:: 2. 全流程一键运行
python pipeline.py

:: 3. 查看结果
cat output\summary.csv
```

## 目录结构

```
TaxiPredict/
├── pipeline.py                  # 一键运行入口
├── config/default.yaml          # 集中配置
├── requirements/                # 依赖管理
├── taxipredict/                 # Python 包
│   ├── config.py                # 配置加载
│   ├── etl/                     # 数据层（加载 / 聚合 / 天气 / 划分）
│   ├── features/                # 特征工程（构建 / 分析 / 选择）
│   ├── models/                  # 模型适配器（BaseModel + 6 模型）
│   └── outputs/                 # 产出层（评估 / 可视化 / SHAP / 报告）
├── data/                        # 数据文件
├── output/                      # 运行产出
│   ├── summary.csv              # 所有模型指标汇总
│   ├── xgboost/                 # 各模型子目录
│   ├── catboost/
│   └── lightgbm/
├── legacy/                      # 旧代码归档（可参考，不执行）
├── scripts/                     # 工具脚本
└── web/                         # Next.js 看板
```

## 用法

### 全流程

```powershell
python pipeline.py                         # ETL → Features → Analyzer → Train（所有模型）
python pipeline.py --model xgboost         # 只跑 XGBoost
python pipeline.py --step etl              # 只跑 ETL
python pipeline.py --force                 # 强制重新执行
```

### 单模型

```powershell
python pipeline.py --model xgboost,catboost,lightgbm
python pipeline.py --model stid             # 需要原始 Uber CSV
python pipeline.py --model gru              # 需要 TensorFlow
python pipeline.py --model lightgcn         # 需要 PyTorch
```

### 特征模式

在 `config/default.yaml` 中设置 `feature_mode`：

| 模式 | 说明 |
|------|------|
| `raw_all` | 全部数值特征（默认） |
| `top_k` | 按训练集相关性取 top-k |
| `ts_topk` | 加时序特征 + 防泄露 + top-k |

## 模型效果

| 模型 | test_MAE | test_RMSE | test_R² |
|------|----------|-----------|---------|
| CatBoost | 0.4001 | 0.7013 | 0.278 |
| LightGBM | 0.4015 | 0.7022 | 0.276 |
| XGBoost | 0.4028 | 0.7042 | 0.272 |
| STID | ~0.45 | ~0.80 | ~0.93 （需要原始 CSV + GPU）|

## 模型产出目录

`output/<model>/` 包含：
- `metrics.json` — 训练/测试指标
- `region_metrics.csv` — 每个 geohash 的独立评估
- `top_regions_actual_vs_pred_simple.csv` — Top-N 区域预测对照
- `plots_top_regions/*.png` — 逐区域对比图
- `report.md` — Markdown 实验报告
- `shap/` — SHAP 特征重要性（需 `pip install shap>=0.46`）
- `model.pkl` — 训练好的模型

## 配置

所有可配置参数在 `config/default.yaml`，支持：

```yaml
data:        # 数据路径
pipeline:    # 聚合方式、测试集划分
features:    # 滞后步数、滚动窗口
models:      # 每个模型的超参数
output:      # 产出开关（shap、report、plots）
```

自定义配置：

```powershell
python pipeline.py --config config/my_config.yaml
```

## 数据

项目使用 [Uber 2014 年 5-6 月纽约行程数据](https://github.com/fivethirtyeight/uber-tlc-foil-response) 和 [NOAA 中央公园天气数据](https://www.ncdc.noaa.gov/cdo-web/)。

原始 CSV 文件需放入 `data/raw/`，预处理脚本会自动处理。

已有中间 CSV 在 `legacy/new/may14/` 和 `legacy/new/jun14/`，ETL 会自动复用。

## 数据流

```
原始 Uber CSV → ETL（清洗 + 聚合 + 天气融合）→ 特征工程 → 模型训练 → 评估报告
```

## SHAP 说明

如需特征可解释性分析：

```powershell
pip install "shap>=0.46"
python pipeline.py --model xgboost --force
```

> **注意**：xgboost 2.x 需要 shap >=0.46，旧版 shap 会报 `could not convert string to float`。

## 旧代码

旧版代码（重构前）位于 `legacy/`，仍可直接运行：

```powershell
python legacy/new/models/xgboostEMA.py --inputs ... --shap
python legacy/new_model/stid_taxi_demand.py --grid-size 0.02
```

新代码 (`taxipredict/`) 已覆盖其核心功能，`legacy/` 仅供对照和参考。

## 环境要求

- Python 3.10+
- 推荐 conda 环境管理

| 依赖 | 用途 |
|------|------|
| requirements/base.txt | 基础（pandas、numpy、scikit-learn 等） |
| requirements/tree.txt | 树模型（xgboost、lightgbm、catboost、shap）|
| requirements/deep.txt | 深度学习（torch、tensorflow，可选） |
