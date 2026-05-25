# legacy/ — 旧代码归档

此目录包含 **TaxiPredict** 的旧版代码，已被新架构 (`taxipredict/` + `pipeline.py`) 取代。

## 各部分说明

### `data_preprocess/`
- **功能**：早期数据预处理，基于 0.01° 网格聚合，融合天气
- **状态**：⛔ 已弃用，被 `process_uber_data.py` 替代
- **仍可运行**：`python legacy/data_preprocess/taxi_data.py`
- **注意**：需要原始 Uber CSV 在 `data_preprocess/data/` 下

### `new/`
- **功能**：geohash 驱动的管线，含特征工程 + 5 个模型
- **状态**：✅ 功能完整，已被 `taxipredict/` + `pipeline.py` 覆盖
- **仍可运行**：
  ```powershell
  python legacy/new/models/xgboostEMA.py --inputs ... --feature-mode raw_all
  python legacy/new/models/CatBoost.py ...（等）
  python legacy/new/features/featureCorr_Entro.py --input ...
  ```
- **新代码已覆盖**：ETL → `taxipredict/etl/`，特征 → `taxipredict/features/`，模型 → `taxipredict/models/`

### `new_model/`
- **功能**：STID（时空身份嵌入模型），当前最佳模型（R² ≈ 0.93）
- **状态**：✅ 功能完整，通过 `taxipredict/models/stid_model.py` 包装调用
- **仍可运行**：
  ```powershell
  python legacy/new_model/stid_taxi_demand.py --grid-size 0.02 --top-regions 30 --epochs 40 --device cpu
  ```
- **注意**：需要原始 Uber CSV（`data_preprocess/data/uber-raw-data-*.csv`）

## 新旧对照表

| 旧代码路径 | 新代码路径 |
|-----------|-----------|
| `new/process_uber_data.py` | `taxipredict/etl/aggregate.py` |
| `new/features/augment_taxi_with_weather_hourly.py` | `taxipredict/etl/weather.py` |
| `new/features/make_xgb_features.py` | `taxipredict/features/builder.py` |
| `new/features/featureCorr_Entro.py` | `taxipredict/features/analyzer.py` |
| `new/models/xgboostEMA.py` | `taxipredict/models/xgboost_model.py` |
| `new/models/CatBoost.py` | `taxipredict/models/catboost_model.py` |
| `new/models/LightGBM.py` | `taxipredict/models/lightgbm_model.py` |
| `new/models/LightGCN.py` | `taxipredict/models/`（待适配）|
| `new/models/GRU.py` | `taxipredict/models/`（待适配）|
| `new_model/stid_taxi_demand.py` | `taxipredict/models/stid_model.py` |

## 何时会用到 legacy 代码

- 验证新代码产出与旧代码一致
- 运行 GRU / LightGCN（尚未适配到新架构）
- 通过 `stid_model.py` 包装器调 STID
- 参考原有实现细节
