import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================
# 1. 读取数据
# ==========================================

print("读取数据...")

may_df = pd.read_csv(
    "../may14/taxi_prediction_hourly_with_weather.csv"
)

jun_df = pd.read_csv(
    "../jun14/taxi_prediction_hourly_with_weather.csv"
)

# ==========================================
# 2. 划分训练集 / 测试集
# ==========================================

print("划分训练测试集...")

# 6月前3周
jun_train_df = jun_df[
    jun_df['day'] <= 22
]

# 训练集
train_df = pd.concat(
    [may_df, jun_train_df],
    ignore_index=True
)

# 测试集：6月最后一周
test_df = jun_df[
    jun_df['day'] >= 23
]

target_col = 'pickups'

# ==========================================
# 3. 构建高级特征
# ==========================================

print("构建高级特征...")

# ---------- 排序 ----------
sort_cols = [
    'year',
    'month',
    'day'
]

train_df = train_df.sort_values(sort_cols)
test_df = test_df.sort_values(sort_cols)

# ==========================================
# 邻近区域特征
# ==========================================

train_df['geo_prefix'] = (
    train_df['geohash']
    .astype(str)
    .str[:5]
)

test_df['geo_prefix'] = (
    test_df['geohash']
    .astype(str)
    .str[:5]
)

# ==========================================
# 历史窗口特征
# ==========================================

windows = [1, 3, 6, 12, 24, 48, 168]

for w in windows:

    train_df[f'pickup_prev{w}'] = (
        train_df
        .groupby('geohash')[target_col]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(w)
            .mean()
        )
    )

    test_df[f'pickup_prev{w}'] = (
        test_df
        .groupby('geohash')[target_col]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(w)
            .mean()
        )
    )

# ==========================================
# 区域统计特征
# ==========================================

geo_mean = (
    train_df
    .groupby('geohash')[target_col]
    .mean()
)

train_df['geo_mean_pickups'] = (
    train_df['geohash']
    .map(geo_mean)
)

test_df['geo_mean_pickups'] = (
    test_df['geohash']
    .map(geo_mean)
)

# ==========================================
# 节假日组合特征
# ==========================================

train_df['holiday_peak'] = (
    train_df['is_holiday']
    * train_df['is_peak']
)

test_df['holiday_peak'] = (
    test_df['is_holiday']
    * test_df['is_peak']
)

# ==========================================
# 天气组合特征
# ==========================================

train_df['bad_weather'] = (
    train_df['is_rain']
    + train_df['is_snow']
    + train_df['is_fog']
)

test_df['bad_weather'] = (
    test_df['is_rain']
    + test_df['is_snow']
    + test_df['is_fog']
)

# ==========================================
# 高峰 + 坏天气
# ==========================================

train_df['peak_bad_weather'] = (
    train_df['is_peak']
    * train_df['bad_weather']
)

test_df['peak_bad_weather'] = (
    test_df['is_peak']
    * test_df['bad_weather']
)

# ==========================================
# 下雨高峰组合
# ==========================================

train_df['rain_peak'] = (
    train_df['is_rain']
    * train_df['is_peak']
)

test_df['rain_peak'] = (
    test_df['is_rain']
    * test_df['is_peak']
)

# ==========================================
# 周末高峰组合
# ==========================================

train_df['weekend_peak'] = (
    train_df['weekend']
    * train_df['is_peak']
)

test_df['weekend_peak'] = (
    test_df['weekend']
    * test_df['is_peak']
)

# ==========================================
# 缺失值处理
# ==========================================

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

# ==========================================
# 4. 特征列
# ==========================================

feature_cols = [
    c for c in train_df.columns
    if c != target_col
]

X_train = train_df[feature_cols].copy()

# log变换（提升预测效果）
y_train = np.log1p(
    train_df[target_col]
)

X_test = test_df[feature_cols].copy()

y_test = test_df[target_col]

# ==========================================
# 5. 类别特征编码
# ==========================================

print("编码类别特征...")

categorical_cols = [
    'time_cat',
    'day_cat',
    'geohash',
    'geo_prefix'
]

for col in categorical_cols:

    le = LabelEncoder()

    le.fit(
        pd.concat([
            X_train[col],
            X_test[col]
        ]).astype(str)
    )

    X_train[col] = le.transform(
        X_train[col].astype(str)
    )

    X_test[col] = le.transform(
        X_test[col].astype(str)
    )

# ==========================================
# 6. LightGBM 数据集
# ==========================================

train_data = lgb.Dataset(
    X_train,
    label=y_train
)

test_data = lgb.Dataset(
    X_test,
    label=np.log1p(y_test),
    reference=train_data
)

# ==========================================
# 7. LightGBM 参数
# ==========================================

params = {

    'objective': 'regression',

    'metric': 'rmse',

    'boosting_type': 'gbdt',

    'learning_rate': 0.02,

    'num_leaves': 31,

    'max_depth': 8,

    'min_data_in_leaf': 120,

    'feature_fraction': 0.7,

    'bagging_fraction': 0.7,

    'bagging_freq': 5,

    'lambda_l1': 2.0,

    'lambda_l2': 2.0,

    'verbose': -1
}

# ==========================================
# 8. 模型训练
# ==========================================

print("开始训练 LightGBM...")

model = lgb.train(

    params,

    train_data,

    num_boost_round=2000,

    valid_sets=[test_data],

    callbacks=[

        lgb.early_stopping(
            stopping_rounds=100
        ),

        lgb.log_evaluation(
            period=50
        )
    ]
)

# ==========================================
# 9. 预测
# ==========================================

print("开始预测...")

y_pred = model.predict(
    X_test,
    num_iteration=model.best_iteration
)

# 反log变换
y_pred = np.expm1(y_pred)

# ==========================================
# 10. 模型评估
# ==========================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n============================")
print("模型评估结果")
print("============================")

print("RMSE:", rmse)
print("MAE :", mae)
print("R^2 :", r2)

print("============================\n")

# ==========================================
# 11. 预测结果可视化
# ==========================================

print("生成预测图...")

plt.figure(figsize=(16, 6))

plot_size = 1000

plt.plot(
    y_test.values[:plot_size],
    label='Actual',
    linewidth=2
)

plt.plot(
    y_pred[:plot_size],
    label='Predicted',
    linewidth=2
)

plt.title(
    'LightGBM: Actual vs Predicted Pickups',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    'Samples',
    fontsize=14
)

plt.ylabel(
    'Pickups',
    fontsize=14
)

plt.legend(
    fontsize=12
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "../jun14/lightgbm_actual_vs_pred_last_week.png",
    dpi=300
)

plt.show()

# ==========================================
# 12. 特征重要性
# ==========================================

print("生成特征重要性图...")

importance = pd.DataFrame({

    'feature': X_train.columns,

    'importance': model.feature_importance()

})

importance = importance.sort_values(
    by='importance',
    ascending=False
).head(20)

plt.figure(figsize=(12, 8))

plt.barh(
    importance['feature'],
    importance['importance']
)

plt.gca().invert_yaxis()

plt.title(
    'Top 20 Feature Importance',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    'Importance',
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "../jun14/lightgbm_feature_importance.png",
    dpi=300
)

plt.show()

print("训练完成！")