import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

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
# 2. 划分训练集和测试集
# ==========================================

print("划分训练测试集...")

# 训练集：5月 + 6月前3周
jun_train_df = jun_df[
    jun_df['day'] <= 22
].copy()

train_df = pd.concat(
    [may_df, jun_train_df],
    ignore_index=True
)

# 测试集：6月最后一周
test_df = jun_df[
    jun_df['day'] >= 23
].copy()

target_col = 'pickups'

# ==========================================
# 3. 构建高级特征
# ==========================================

print("构建高级特征...")

for df in [train_df, test_df]:

    # 高峰 + 周末
    df['peak_weekend'] = (
        df['is_peak'] * df['weekend']
    )

    # 节假日 + 高峰
    df['holiday_peak'] = (
        df['is_holiday'] * df['is_peak']
    )

    # 下雨 + 高峰
    df['rain_peak'] = (
        df['is_rain'] * df['is_peak']
    )

    # 温度 + 湿度
    df['temp_rhum'] = (
        df['temp'] * df['rhum']
    )

    # 温度 + 风速
    df['temp_wspd'] = (
        df['temp'] * df['wspd']
    )

# ==========================================
# 4. 邻近区域聚合特征
# ==========================================

print("构建邻近区域聚合特征...")

# geohash 前缀
train_df['geo_prefix'] = (
    train_df['geohash']
    .astype(str)
    .str[:4]
)

test_df['geo_prefix'] = (
    test_df['geohash']
    .astype(str)
    .str[:4]
)

# 邻近区域平均订单量
geo_mean = (
    train_df.groupby('geo_prefix')['pickups']
    .mean()
)

train_df['geo_avg_pickups'] = (
    train_df['geo_prefix']
    .map(geo_mean)
)

test_df['geo_avg_pickups'] = (
    test_df['geo_prefix']
    .map(geo_mean)
)

# 缺失值填充
test_df['geo_avg_pickups'] = (
    test_df['geo_avg_pickups']
    .fillna(train_df['pickups'].mean())
)

# ==========================================
# 5. 历史滑动窗口特征
# ==========================================

print("构建历史窗口特征...")

train_df = train_df.sort_values(
    ['geohash', 'day', 'time_cat']
)

test_df = test_df.sort_values(
    ['geohash', 'day', 'time_cat']
)

# 前1小时订单量
train_df['pickup_lag1'] = (
    train_df.groupby('geohash')['pickups']
    .shift(1)
    .fillna(0)
)

# 前3小时移动平均
train_df['pickup_prev3'] = (
    train_df.groupby('geohash')['pickups']
    .transform(
        lambda x:
        x.shift(1)
        .rolling(3)
        .mean()
    )
    .fillna(0)
)

# 测试集使用训练集均值近似
test_df['pickup_lag1'] = (
    train_df['pickup_lag1'].mean()
)

test_df['pickup_prev3'] = (
    train_df['pickup_prev3'].mean()
)

# ==========================================
# 6. 特征列表
# ==========================================

feature_cols = [
    c for c in train_df.columns
    if c != target_col
]

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col]

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col]

# ==========================================
# 7. 编码类别特征
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

    all_values = pd.concat([
        X_train[col].astype(str),
        X_test[col].astype(str)
    ])

    le.fit(all_values)

    X_train[col] = le.transform(
        X_train[col].astype(str)
    )

    X_test[col] = le.transform(
        X_test[col].astype(str)
    )

# ==========================================
# 8. log1p目标变换
# ==========================================

print("进行目标变换...")

y_train_log = np.log1p(y_train)

# ==========================================
# 9. CatBoost训练
# ==========================================

print("开始训练 CatBoost...")

model = CatBoostRegressor(

    iterations=800,

    learning_rate=0.05,

    depth=6,

    loss_function='RMSE',

    eval_metric='RMSE',

    random_seed=42,

    l2_leaf_reg=8,

    subsample=0.8,

    random_strength=2,

    bagging_temperature=1,

    verbose=100
)

model.fit(
    X_train,
    y_train_log,

    eval_set=(
        X_test,
        np.log1p(y_test)
    ),

    use_best_model=True
)

# ==========================================
# 10. 开始预测
# ==========================================

print("开始预测...")

y_pred_log = model.predict(X_test)

# 反变换
y_pred = np.expm1(y_pred_log)

# 防止负值
y_pred = np.maximum(y_pred, 0)

# ==========================================
# 11. 模型评估
# ==========================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n============================")
print("CatBoost 模型评估结果")
print("============================")
print("RMSE:", rmse)
print("MAE :", mae)
print("R^2 :", r2)
print("============================")

# ==========================================
# 12. 生成预测结果图
# ==========================================

print("生成预测图...")

plt.figure(figsize=(16, 6))

sample_num = 1000

plt.plot(
    y_test.values[:sample_num],
    label='Actual',
    linewidth=2
)

plt.plot(
    y_pred[:sample_num],
    label='Predicted',
    linewidth=2
)

plt.title(
    'CatBoost: Actual vs Predicted Pickups',
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

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "catboost_actual_vs_pred.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==========================================
# 13. 特征重要性图
# ==========================================

print("生成特征重要性图...")

importance = model.get_feature_importance()

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
})

feature_importance = feature_importance.sort_values(
    by='importance',
    ascending=False
).head(15)

plt.figure(figsize=(12, 8))

plt.barh(
    feature_importance['feature'],
    feature_importance['importance']
)

plt.title(
    'Top 15 Feature Importance (CatBoost)',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    'Importance',
    fontsize=14
)

plt.ylabel(
    'Features',
    fontsize=14
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "catboost_feature_importance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("训练完成！")