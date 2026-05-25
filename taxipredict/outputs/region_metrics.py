"""按区域（geohash）计算评估指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_region_metrics(
    pred_df: pd.DataFrame,
    target_col: str = "pickups",
    group_col: str = "geohash",
) -> pd.DataFrame:
    """按区域分组计算 MAE / RMSE / R²。

    输入 pred_df 需包含：
    - group_col 列：区域标识（如 geohash）
    - target_col 列：真实值
    - pred_pickups 列：预测值
    """
    if group_col not in pred_df.columns:
        raise ValueError(f"pred_df 缺少区域列: {group_col}")

    rows = []
    for region, group_df in pred_df.groupby(group_col, sort=True):
        y_true = group_df[target_col].values
        y_pred = group_df["pred_pickups"].values

        if len(y_true) == 0:
            continue

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan

        rows.append({
            group_col: region,
            "test_rows": len(y_true),
            "mae": mae,
            "rmse": rmse,
            "r2": r2 if not np.isnan(r2) else np.nan,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("r2", ascending=False, na_position="last").reset_index(drop=True)
    return result
