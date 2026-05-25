"""LightGBM 模型适配器。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxipredict.features.selector import select_features
from taxipredict.models.base import BaseModel

try:
    from lightgbm import LGBMRegressor
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    LGBMRegressor = None


class LightGBMModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._model = None
        self._feature_cols = []
        self._target_col = "pickups"

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm 未安装，请运行: pip install lightgbm")

        lgb_cfg = self.cfg.get("models", {}).get("lightgbm", {})
        feature_mode = lgb_cfg.get("feature_mode", "raw_all")
        feature_k = lgb_cfg.get("feature_k", 20)

        self._feature_cols, _train, _test = select_features(
            train_df=train_df,
            test_df=test_df,
            mode=feature_mode,
            k=feature_k,
            target_col=self._target_col,
        )

        X_train = _train[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = _train[self._target_col]
        X_test = _test[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = _test[self._target_col]

        self._model = LGBMRegressor(
            n_estimators=lgb_cfg.get("n_estimators", 500),
            num_leaves=lgb_cfg.get("num_leaves", 31),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            subsample=lgb_cfg.get("subsample", 0.8),
            colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds=lgb_cfg.get("early_stopping_rounds", 20),
            random_state=42,
            verbose=-1,
        )

        self._model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        train_pred = np.maximum(self._model.predict(X_train), 0)
        test_pred = np.maximum(self._model.predict(X_test), 0)

        self._pred_df = _test[[self._target_col, "datetime", "geohash"]].copy()
        self._processed_test = _test
        self._pred_df["pred_pickups"] = test_pred
        self._pred_df["error"] = self._pred_df["pred_pickups"] - self._pred_df[self._target_col]
        self._pred_df["abs_error"] = self._pred_df["error"].abs()

        return {
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_mae": float(mean_absolute_error(y_test, test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "test_r2": float(r2_score(y_test, test_pred)),
            "feature_count": len(self._feature_cols),
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型未训练")
        X = df[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        return np.maximum(self._model.predict(X), 0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "features": self._feature_cols}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMModel":
        data = joblib.load(path)
        model = cls.__new__(cls)
        model._model = data["model"]
        model._feature_cols = data["features"]
        return model
