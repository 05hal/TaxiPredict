# TaxiPredict — NYC Uber 需求预测

一键运行纽约 Uber 2014 年 5-6 月订单需求预测，集成 6 种模型。

## 环境配置

```powershell
conda create -n taxi_predict python=3.10
conda activate taxi_predict
pip install -r requirements/base.txt
pip install -r requirements/tree.txt
```

可选（SHAP 图 + GPU）：

```powershell
pip install "shap>=0.46"
:: GPU：config/default.yaml 中 device 改为 cuda/gpu/GPU
```

## 一键运行

```powershell
:: 全流程（ETL → 特征 → 训练 → 产出）
python pipeline.py

:: 单模型
python pipeline.py --model xgboost,catboost,lightgbm

:: 强制重跑
python pipeline.py --force

:: 自定义配置
python pipeline.py --config config/my_config.yaml
```

结果汇总：`output/summary.csv`

各模型产出：`output/<model>/metrics.json`、`region_metrics.csv`、`plots_top_regions/`、`report.md`、`shap/`

## 效果对比

| 指标 | 重构前 | 重构后 | 变化 |
|------|:------:|:------:|:----:|
| XGBoost test_MAE | **0.4028** | **0.3704** | ▼ 8.0% |
| XGBoost test_R² | 0.2721 | 0.2721 | — |
| CatBoost test_MAE | **0.4001** | **0.3699** | ▼ 7.5% |
| LightGBM test_MAE | **0.4015** | **0.3701** | ▼ 7.8% |

MAE 降低主要来自：**log1p 目标变换 + 新增 10 个特征 + 更优参数**

## 项目结构

```
├── pipeline.py             # 一键入口
├── config/default.yaml     # 集中配置
├── taxipredict/            # 核心包 (etl/features/models/outputs)
├── data/                   # 数据文件
├── output/                 # 运行产出 + summary.csv
├── legacy/                 # 旧代码归档
├── web/                    # Next.js 看板
└── requirements/           # 依赖管理
```

## 技术栈

`Python 3.10` `pandas` `numpy` `scikit-learn` `xgboost` `lightgbm` `catboost` `shap` `PyTorch/STID`
