首先解压jun14 may14到当前目录

1.处理天气数据：
五月
python features\\augment\_taxi\_with\_weather\_hourly.py ^
--taxi-csv may14\\taxi\_prediction\_style\_aggregated.csv ^
--weather-csv LCD\_USW00094728\_2014.csv ^
--out-csv may14\\taxi\_prediction\_hourly\_with\_weather.csv

六月
python features\\augment\_taxi\_with\_weather\_hourly.py ^
--taxi-csv jun14\\taxi\_prediction\_style\_aggregated.csv ^
--weather-csv LCD\_USW00094728\_2014.csv ^
--out-csv jun14\\taxi\_prediction\_hourly\_with\_weather.csv

2.运行特征报告：
python features\\featureCorr\_Entro.py --input may14\\taxi\_prediction\_hourly\_with\_weather.csv
python features\\featureCorr\_Entro.py --input jun14\\taxi\_prediction\_hourly\_with\_weather.csv

3.运行示例预测程序，输出shap分析
python models\\xgboostEMA.py ^
--inputs may14\\taxi\_prediction\_hourly\_with\_weather.csv jun14\\taxi\_prediction\_hourly\_with\_weather.csv ^
--feature-mode raw\_all ^
--shap

或者加时序特征：

python models\\xgboostEMA.py ^
--inputs may14\\taxi\_prediction\_hourly\_with\_weather.csv jun14\\taxi\_prediction\_hourly\_with\_weather.csv ^
--feature-mode ts\_topk ^
--k 30 ^
--shap

目前效果不太好

## 模型环境（conda `taxi-models`）

```bat
REM 首次安装（在 new 目录）
setup_models_env.bat

REM 若 GRU / LightGCN 报 torch._prims_common 或 tensorflow 导入错误（残缺安装）
repair_deep_models.bat
```

| 模型 | 依赖 | 运行示例 |
|------|------|----------|
| LightGBM / CatBoost | `requirements-models.txt` | `python models\LightGBM.py --inputs may14\xgb_dense_features.csv jun14\xgb_dense_features.csv --feature-mode raw_all` |
| GRU | TensorFlow 2.16 | `python models\GRU.py --inputs may14\xgb_dense_features.csv jun14\xgb_dense_features.csv --feature-mode raw_all` |
| LightGCN | PyTorch 2.4.1 CPU | `python models\LightGCN.py --inputs may14\xgb_dense_features.csv jun14\xgb_dense_features.csv --feature-mode raw_all` |

验证深度学习依赖：`python verify_deep_models.py`

