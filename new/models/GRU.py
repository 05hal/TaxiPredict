import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.callbacks import (
    EarlyStopping
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
# 2. 划分训练测试集
# ==========================================

print("划分训练测试集...")

jun_train_df = jun_df[
    jun_df['day'] <= 22
].copy()

train_df = pd.concat(
    [may_df, jun_train_df],
    ignore_index=True
)

test_df = jun_df[
    jun_df['day'] >= 23
].copy()

# ==========================================
# 3. 抽样训练数据（减少内存）
# ==========================================

print("抽样训练数据...")

train_df = train_df.sample(
    frac=0.15,
    random_state=42
).reset_index(drop=True)

# ==========================================
# 4. 目标列
# ==========================================

target_col = 'pickups'

# ==========================================
# 5. 构建历史 pickup 特征
# ==========================================

print("构建历史 pickup 特征...")

train_df = train_df.sort_values(
    by=['geohash', 'day']
)

test_df = test_df.sort_values(
    by=['geohash', 'day']
)

train_df['pickup_lag1'] = (
    train_df
    .groupby('geohash')[target_col]
    .shift(1)
)

train_df['pickup_lag3'] = (
    train_df
    .groupby('geohash')[target_col]
    .shift(3)
)

test_df['pickup_lag1'] = (
    test_df
    .groupby('geohash')[target_col]
    .shift(1)
)

test_df['pickup_lag3'] = (
    test_df
    .groupby('geohash')[target_col]
    .shift(3)
)

# 缺失值填充
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# ==========================================
# 6. 高级特征工程
# ==========================================

print("构建高级特征...")

for df in [train_df, test_df]:

    # 高峰+周末
    df['peak_weekend'] = (
        df['is_peak'] * df['weekend']
    )

    # 节假日+高峰
    df['holiday_peak'] = (
        df['is_holiday'] * df['is_peak']
    )

    # 下雨+高峰
    df['rain_peak'] = (
        df['is_rain'] * df['is_peak']
    )

    # 温度湿度组合
    df['temp_rhum'] = (
        df['temp'] * df['rhum']
    )

    # 温度风速组合
    df['temp_wspd'] = (
        df['temp'] * df['wspd']
    )

# ==========================================
# 7. 编码 geohash
# ==========================================

print("编码 geohash...")

le = LabelEncoder()

all_geo = pd.concat([
    train_df['geohash'].astype(str),
    test_df['geohash'].astype(str)
])

le.fit(all_geo)

train_df['geohash'] = le.transform(
    train_df['geohash'].astype(str)
)

test_df['geohash'] = le.transform(
    test_df['geohash'].astype(str)
)

# ==========================================
# 8. 特征列
# ==========================================

feature_cols = [

    'year',
    'month',
    'day',

    'time_cos',
    'time_sin',

    'day_cos',
    'day_sin',

    'weekend',

    'geohash',

    'latitude',
    'logitude',

    'temp',
    'rhum',
    'wspd',

    'is_precip',
    'is_rain',
    'is_snow',
    'is_fog',

    'is_holiday',
    'is_peak',

    'peak_weekend',
    'holiday_peak',
    'rain_peak',

    'temp_rhum',
    'temp_wspd',

    # 历史特征
    'pickup_lag1',
    'pickup_lag3'
]

# ==========================================
# 9. 处理异常值
# ==========================================

print("处理异常值...")

train_df.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)

test_df.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# ==========================================
# 10. 标准化
# ==========================================

print("标准化数据...")

scaler = StandardScaler()

train_features = scaler.fit_transform(
    train_df[feature_cols]
).astype(np.float32)

test_features = scaler.transform(
    test_df[feature_cols]
).astype(np.float32)

train_target = train_df[
    target_col
].values.astype(np.float32)

test_target = test_df[
    target_col
].values.astype(np.float32)

# ==========================================
# 11. 构建时间窗口
# ==========================================

print("构建时间序列窗口...")

WINDOW_SIZE = 24

def create_sequences(features, target, window_size):

    X = []
    y = []

    for i in range(window_size, len(features)):

        X.append(
            features[i-window_size:i]
        )

        y.append(
            target[i]
        )

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(
    train_features,
    train_target,
    WINDOW_SIZE
)

X_test, y_test = create_sequences(
    test_features,
    test_target,
    WINDOW_SIZE
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ==========================================
# 12. 构建 GRU 模型
# ==========================================

print("构建 GRU 模型...")

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    GRU(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    GRU(32),

    Dropout(0.2),

    Dense(1)
])

# ==========================================
# 13. 编译模型
# ==========================================

model.compile(

    optimizer='adam',

    loss='mse',

    metrics=['mae']
)

model.summary()

# ==========================================
# 14. EarlyStopping
# ==========================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=3,

    restore_best_weights=True
)

# ==========================================
# 15. 开始训练
# ==========================================

print("开始训练 GRU...")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=15,

    batch_size=128,

    callbacks=[early_stop],

    verbose=1
)

# ==========================================
# 16. 预测
# ==========================================

print("开始预测...")

y_pred = model.predict(X_test)

y_pred = y_pred.flatten()

# 防止负数
y_pred = np.maximum(
    y_pred,
    0
)

# ==========================================
# 17. 模型评估
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
print("GRU 模型评估结果")
print("============================")
print("RMSE:", rmse)
print("MAE :", mae)
print("R^2 :", r2)
print("============================")

# ==========================================
# 18. 预测结果图
# ==========================================

print("生成预测图...")

plt.figure(figsize=(16,6))

sample_num = 1000

plt.plot(
    y_test[:sample_num],
    label='Actual',
    linewidth=2
)

plt.plot(
    y_pred[:sample_num],
    label='Predicted',
    linewidth=2
)

plt.title(
    'GRU: Actual vs Predicted Pickups',
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
    "gru_actual_vs_pred.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==========================================
# 19. Loss曲线
# ==========================================

print("生成 Loss 曲线...")

plt.figure(figsize=(10,6))

plt.plot(
    history.history['loss'],
    label='Train Loss',
    linewidth=2
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss',
    linewidth=2
)

plt.title(
    'GRU Training Loss',
    fontsize=18,
    fontweight='bold'
)

plt.xlabel(
    'Epoch',
    fontsize=14
)

plt.ylabel(
    'Loss',
    fontsize=14
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "gru_loss_curve.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("训练完成！")