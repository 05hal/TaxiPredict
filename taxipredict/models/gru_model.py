"""GRU 模型适配器——TensorFlow 门控循环单元。

将表格数据转为滑动窗口，用 GRU 层训练预测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from taxipredict.features.selector import select_features
from taxipredict.models.base import BaseModel

try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


def _create_sequences(features, target, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)


class GRUModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._model = None
        self._feature_cols = []
        self._target_col = "pickups"
        self._window_size = 48
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，GRU 需要 tensorflow>=2.16")

        gru_cfg = self.cfg.get("models", {}).get("gru", {})
        self._window_size = gru_cfg.get("input_len", 48)

        # 特征选择
        self._feature_cols, _train, _test = select_features(
            train_df, test_df, mode="raw_all", target_col=self._target_col,
        )

        # 构建序列
        train_seq, train_target = _create_sequences(
            _train[self._feature_cols].fillna(0).values,
            _train[self._target_col].values,
            self._window_size,
        )
        test_seq, test_target = _create_sequences(
            _test[self._feature_cols].fillna(0).values,
            _test[self._target_col].values,
            self._window_size,
        )

        train_seq_2d = train_seq.reshape(-1, train_seq.shape[2])
        self._scaler_X.fit(train_seq_2d)
        train_seq = self._scaler_X.transform(train_seq.reshape(-1, train_seq.shape[2])).reshape(train_seq.shape)
        test_seq = self._scaler_X.transform(test_seq.reshape(-1, test_seq.shape[2])).reshape(test_seq.shape)

        self._scaler_y.fit(train_target.reshape(-1, 1))

        model = Sequential([
            Input(shape=(self._window_size, len(self._feature_cols))),
            GRU(gru_cfg.get("hidden_units", [64])[0], return_sequences=True),
            Dropout(gru_cfg.get("dropout", 0.2)),
            GRU(gru_cfg.get("hidden_units", [64, 32])[-1] if len(gru_cfg.get("hidden_units", [64, 32])) > 1 else 32),
            Dropout(gru_cfg.get("dropout", 0.2)),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")

        es = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(
            train_seq, self._scaler_y.transform(train_target.reshape(-1, 1)),
            validation_data=(test_seq, self._scaler_y.transform(test_target.reshape(-1, 1))),
            epochs=gru_cfg.get("epochs", 50),
            batch_size=gru_cfg.get("batch_size", 32),
            callbacks=[es],
            verbose=0,
        )
        self._model = model

        train_pred = self._scaler_y.inverse_transform(model.predict(train_seq, verbose=0)).flatten()
        test_pred = self._scaler_y.inverse_transform(model.predict(test_seq, verbose=0)).flatten()

        self._pred_df = _test.iloc[self._window_size:][[self._target_col, "datetime", "geohash"]].copy()
        self._pred_df["pred_pickups"] = np.maximum(test_pred, 0)
        self._pred_df["error"] = self._pred_df["pred_pickups"] - self._pred_df[self._target_col]
        self._pred_df["abs_error"] = self._pred_df["error"].abs()
        self._processed_test = _test

        return {
            "train_mae": float(mean_absolute_error(train_target, np.maximum(train_pred, 0))),
            "train_rmse": float(np.sqrt(mean_squared_error(train_target, np.maximum(train_pred, 0)))),
            "train_r2": float(r2_score(train_target, np.maximum(train_pred, 0))),
            "test_mae": float(mean_absolute_error(test_target, np.maximum(test_pred, 0))),
            "test_rmse": float(np.sqrt(mean_squared_error(test_target, np.maximum(test_pred, 0)))),
            "test_r2": float(r2_score(test_target, np.maximum(test_pred, 0))),
            "feature_count": len(self._feature_cols),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("GRU predict 需要序列构建，请使用 train() 中的 _test 评估")

    def save(self, path: str | Path) -> None:
        if self._model is not None:
            self._model.save(str(path).replace(".pkl", ".keras"))

    @classmethod
    def load(cls, path: str | Path) -> "GRUModel":
        model = cls.__new__(cls)
        model._model = None
        return model
