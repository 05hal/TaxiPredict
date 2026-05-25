"""BaseModel 抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """所有预测模型的统一接口。

    生命周期：
        model = ModelClass(cfg)     # 构造
        metrics = model.train(train_df, test_df)  # 训练 + 评估
        model.predict(test_df)      # 预测
        model.save(path)            # 保存到磁盘
        model = ModelClass.load(path)  # 从磁盘加载
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._model = None
        self._feature_cols: list[str] = []
        self._pred_df = None
        self._processed_test = None
        self._ranking_df = None

    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """训练模型并返回指标字典。

        返回示例：{"train_mae": ..., "test_mae": ..., "test_r2": ...}
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """对输入 DataFrame 执行预测，返回预测值数组。"""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """将模型保存到磁盘。"""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """从磁盘加载模型。"""
        ...

    def decode_target(self, y_pred: np.ndarray, target_col: str) -> np.ndarray:
        """若目标列使用 log1p 变换，解码回原始量纲。"""
        if target_col == "y_log1p":
            return np.maximum(np.expm1(y_pred), 0)
        return y_pred

    def _select_features(
        self, df: pd.DataFrame, target_col: str = "pickups"
    ) -> list[str]:
        """自动选择数值特征列（排除 ID/日期/目标列）。"""
        exclude = {
            target_col, "datetime", "geohash", "latitude", "longitude",
            "time_cat", "day_cat", "split", "is_test", "source_file",
            "year", "month", "day", "hour",
            "y_log1p", "_row_id", "_src",
        }
        return [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
