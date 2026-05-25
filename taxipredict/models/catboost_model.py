"""CatBoost 模型适配器。"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taxipredict.features.selector import select_features
from taxipredict.models.base import BaseModel

try:
    from catboost import CatBoostRegressor
    _CAT_AVAILABLE = True
except ImportError:
    _CAT_AVAILABLE = False
    CatBoostRegressor = None


class CatBoostModel(BaseModel):
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
        if not _CAT_AVAILABLE:
            raise ImportError("catboost 未安装，请运行: pip install catboost")

        cb_cfg = self.cfg.get("models", {}).get("catboost", {})
        feature_mode = cb_cfg.get("feature_mode", "raw_all")
        feature_k = cb_cfg.get("feature_k", 20)
        use_log1p = cb_cfg.get("log1p", False)
        actual_target = "y_log1p" if use_log1p else self._target_col

        self._feature_cols, _train, _test, self._ranking_df = select_features(
            train_df=train_df,
            test_df=test_df,
            mode=feature_mode,
            k=feature_k,
            target_col=actual_target,
        )

        _cat_cols_extra = ["geo_prefix", "time_cat", "day_cat"]
        _cat_cols = []
        for col in _cat_cols_extra:
            if col in _train.columns and col not in self._feature_cols:
                self._feature_cols.append(col)
                _cat_cols.append(col)
            elif col in _train.columns:
                _cat_cols.append(col)

        X_train = _train[self._feature_cols].copy()
        X_train[_cat_cols] = X_train[_cat_cols].astype(str).fillna("") if _cat_cols else X_train[_cat_cols]
        for col in self._feature_cols:
            if col not in _cat_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
        y_train = _train[actual_target]

        X_test = _test[self._feature_cols].copy()
        X_test[_cat_cols] = X_test[_cat_cols].astype(str).fillna("") if _cat_cols else X_test[_cat_cols]
        for col in self._feature_cols:
            if col not in _cat_cols:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(0)
        y_test = _test[actual_target]

        device = cb_cfg.get("device", "CPU").upper()
        self._model = CatBoostRegressor(
            iterations=cb_cfg.get("iterations", 800),
            learning_rate=cb_cfg.get("learning_rate", 0.05),
            depth=cb_cfg.get("depth", 6),
            l2_leaf_reg=cb_cfg.get("l2_leaf_reg", 8),
            subsample=cb_cfg.get("subsample", 0.85),
            bootstrap_type="Bernoulli" if device == "GPU" else None,
            task_type=device,
            loss_function="RMSE",
            random_seed=42,
            verbose=0,
        )

        if _cat_cols:
            cat_features_indices = [list(X_train.columns).index(c) for c in _cat_cols]
        else:
            cat_features_indices = None

        self._model.fit(X_train, y_train, eval_set=(X_test, y_test),
                        cat_features=cat_features_indices,
                        use_best_model=True, verbose=False)

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
    def load(cls, path: str | Path) -> "CatBoostModel":
        data = joblib.load(path)
        model = cls.__new__(cls)
        model._model = data["model"]
        model._feature_cols = data["features"]
        return model
