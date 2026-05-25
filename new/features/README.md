# make_xgb_features.py

为 XGBoost 训练准备“更稠密”的特征文件，用于缓解 `taxi_prediction_hourly_with_weather.csv` 中天气/事件类特征偏稀疏的问题，并强化可预测信号（主要来自客流的时序滞后与滚动统计）。

## 输入

脚本默认读取两份已完成对齐的特征数据：

- `d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\taxi_prediction_hourly_with_weather.csv`
- `d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\taxi_prediction_hourly_with_weather.csv`

要求至少包含这些列（用于构造时间键与标签）：

- 时间：`year, month, day, time_cat`
- 空间：`geohash`
- 标签：`pickups`

其余列（天气、节假日、高峰等）存在则使用，不存在会自动降级处理。

## 输出

脚本会分别输出两份文件，并且保持“原始列顺序不变”，新增特征追加在末尾：

- `d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\xgb_dense_features.csv`
- `d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\xgb_dense_features.csv`

## 主要动作：

### 1) 生成 dt 时间戳（内部使用，不写入输出）
从 `year/month/day/time_cat(含分钟)` 解析出 `dt`，用于滚动窗口、滞后特征计算。

### 2) 天气“去稀疏化”特征
把连续/稀疏的天气变量压缩成更容易被树模型学习的特征，例如：

- `precip_level`：降水强度分桶
- `low_vis`：低能见度指示（vis < 8km）
- `high_wind`：强风指示（wspd > 8）
- `temp_bin`：温度分桶
- 交互项：`peak_x_precip`, `peak_x_precip_level`, `weekend_x_precip`, `evening_x_low_vis`
- 缺失指示：`missing_temp`, `missing_vis`, `missing_wspd`, `missing_precip`

### 3) 天气滚动窗口特征（按时间聚合后回填）
先按 `dt` 对全城（同一时刻各 geohash 行）天气取均值，再做滚动统计：

- `precip_3h_sum`, `precip_6h_sum`
- `is_precip_6h_any`
- `temp_3h_delta`, `vis_3h_min`, `wspd_3h_max`

说明：滚动窗口以 30 分钟桶为步长（3 小时=6 个桶，6 小时=12 个桶）。

### 4) 客流时序特征
对每个 `geohash` 按时间排序生成滞后、滚动、相对变化特征：

- 滞后：`pickups_lag1`, `pickups_lag2`, `pickups_lag48`(24h), `pickups_lag336`(7d)
- 滚动均值：`pickups_roll6_mean`(3h), `pickups_roll12_mean`(6h), `pickups_roll48_mean`(24h)
- 长期均值：`pickups_expanding_mean`
- 相对/差分：`pickups_mom1`, `pickups_diff_24h`, `pickups_ratio_roll48`
- 缺失指示：`missing_pickups_lag1`

### 5) 训练目标
- `y_log1p = log(1 + pickups)`：建议作为 XGBoost 回归目标。

### 6) 自动删除常量列（按月输出分别判断）
对每个月输出文件，自动删除“全为同一个值”的列（例如 5/6 月 `is_snow` 常为 0）。
不会删除关键列：`year/month/day/time_cat/geohash/pickups/latitude/longitude` 等。

## 用法

PowerShell 运行：

```powershell
python d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\make_xgb_features.py
```

自定义输入输出：

```powershell
python d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\make_xgb_features.py `
  --may-csv d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\taxi_prediction_hourly_with_weather.csv `
  --jun-csv d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\taxi_prediction_hourly_with_weather.csv `
  --may-out-csv d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\xgb_dense_features.csv `
  --jun-out-csv d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\xgb_dense_features.csv
```

如果你暂时不想生成“客流滞后/滚动”特征：

```powershell
python d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\make_xgb_features.py --no-pickup-lags
```

## 注意事项
- 脚本不负责训练/切分数据集；它只生成特征文件。
- `pickups_lag*` 等滞后特征在序列开头会出现缺失，属于正常现象；你可以在训练时让 XGBoost 直接处理 missing。
- `day_cat` 是字符串列，若用于训练需自行编码（或丢弃，仅用 cos/sin 周期特征）。