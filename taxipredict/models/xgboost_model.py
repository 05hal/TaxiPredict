"""XGBoost 模型适配器。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxipredict.features.selector import select_features
from taxipredict.models.base import BaseModel

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    XGBRegressor = None


class XGBoostModel(BaseModel):
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
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost 未安装，请运行: pip install xgboost")

        xgb_cfg = self.cfg.get("models", {}).get("xgboost", {})
        feature_mode = xgb_cfg.get("feature_mode", "raw_all")
        feature_k = xgb_cfg.get("feature_k", 20)
        use_log1p = xgb_cfg.get("log1p", False)
        actual_target = "y_log1p" if use_log1p else self._target_col

        self._feature_cols, _train, _test, self._ranking_df = select_features(
            train_df=train_df,
            test_df=test_df,
            mode=feature_mode,
            k=feature_k,
            target_col=actual_target,
        )

        X_train = _train[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = _train[actual_target]
        X_test = _test[self._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = _test[actual_target]

        self._model = XGBRegressor(
            n_estimators=xgb_cfg.get("n_estimators", 500),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 20),
            device=xgb_cfg.get("device", "cpu"),
            eval_metric="rmse",
            random_state=42,
            verbosity=0,
        )

        self._model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        train_pred = self.decode_target(np.maximum(self._model.predict(X_train), 0), actual_target)
        test_pred = self.decode_target(np.maximum(self._model.predict(X_test), 0), actual_target)
        y_train_raw = _train[self._target_col]
        y_test_raw = _test[self._target_col]

        self._pred_df = _test[[self._target_col, "datetime", "geohash"]].copy()
        self._pred_df["pred_pickups"] = test_pred
        self._pred_df["error"] = self._pred_df["pred_pickups"] - self._pred_df[self._target_col]
        self._pred_df["abs_error"] = self._pred_df["error"].abs()
        self._processed_test = _test

        return {
            "train_mae": float(mean_absolute_error(y_train_raw, train_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train_raw, train_pred))),
            "train_r2": float(r2_score(y_train_raw, train_pred)),
            "test_mae": float(mean_absolute_error(y_test_raw, test_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test_raw, test_pred))),
            "test_r2": float(r2_score(y_test_raw, test_pred)),
            "feature_count": len(self._feature_cols),
            "log1p": use_log1p,
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
    def load(cls, path: str | Path) -> "XGBoostModel":
        data = joblib.load(path)
        model = cls.__new__(cls)
        model._model = data["model"]
        model._feature_cols = data["features"]
        return model
