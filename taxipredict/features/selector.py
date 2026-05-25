"""特征选择：raw_all / top_k / ts_topk 模式。

ts_topk 模式会先构造时序特征（lag/rolling/EMA/diff），
测试集行用 NaN 填充 observed_pickups 防止未来信息泄露，
然后按训练集相关性取 top-k 特征。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _build_ts_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "pickups",
    group_col: str = "geohash",
) -> pd.DataFrame:
    """构造时序特征，测试集行用 NaN 防泄露。"""
    full = pd.concat([train_df, test_df], ignore_index=True)
    full = full.sort_values([group_col, "datetime"]).reset_index(drop=True)

    # 测试集 observed_pickups 设为 NaN，阻止未来数据泄漏到特征中
    full["observed_pickups_for_ts"] = np.where(
        full.get("is_test", pd.Series(0, index=full.index)) == 1,
        np.nan,
        full[target_col],
    )

    g = full.groupby(group_col, sort=False)["observed_pickups_for_ts"]

    # lag 特征
    for lag in [1, 3, 6, 24]:
        full[f"ts_lag_{lag}"] = g.shift(lag)

    # rolling mean（基于 shift(1) 避免泄漏）
    shifted = g.shift(1)
    for w in [3, 6, 24]:
        full[f"ts_rolling_mean_{w}"] = (
            shifted.groupby(full[group_col])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # EMA
    for span in [3, 6, 24]:
        full[f"ts_ema_{span}"] = (
            shifted.groupby(full[group_col])
            .ewm(span=span, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # diff
    prev1 = g.shift(1)
    prev2 = g.shift(2)
    full["ts_diff_1"] = prev1 - prev2

    # 同小时 lag
    if "time_cat" in full.columns:
        hour = pd.to_numeric(full["time_cat"].astype(str).str.slice(0, 2), errors="coerce")
    else:
        hour = full["datetime"].dt.hour
    full["ts_hour"] = hour
    full["ts_same_hour_lag_1"] = (
        full.groupby([group_col, "ts_hour"])["observed_pickups_for_ts"]
        .shift(1)
    )
    full = full.drop(columns=["ts_hour", "observed_pickups_for_ts"])

    return full


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    temp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(temp) < 2 or temp["x"].nunique() <= 1 or temp["y"].nunique() <= 1:
        return 0.0
    return float(temp["x"].corr(temp["y"]))


def _get_ts_feature_names() -> list[str]:
    return [
        "ts_lag_1", "ts_lag_3", "ts_lag_6", "ts_lag_24",
        "ts_rolling_mean_3", "ts_rolling_mean_6", "ts_rolling_mean_24",
        "ts_ema_3", "ts_ema_6", "ts_ema_24",
        "ts_diff_1", "ts_same_hour_lag_1",
    ]


def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str = "raw_all",
    k: int = 20,
    target_col: str = "pickups",
    group_col: str = "geohash",
    exclude_cols: set[str] | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """执行特征选择，返回 (selected_feature_cols, train_df, test_df)。

    模式：
    - raw_all：使用全部数值特征
    - top_k：按训练集相关性取 top k
    - ts_topk：先构建时序特征，再按训练集相关性取 top k
    """
    _train = train_df.copy()
    _test = test_df.copy()

    if exclude_cols is None:
        exclude_cols = {
            target_col, "datetime", group_col, "latitude", "longitude",
            "time_cat", "day_cat", "split", "is_test", "source_file",
            "year", "month", "day", "hour",
            "y_log1p", "_row_id", "_src",
        }

    # ts_topk 模式：先构建时序特征
    if mode == "ts_topk":
        full = _build_ts_features(_train, _test, target_col, group_col)
        _train = full[full["split"] == "train"].copy()
        _test = full[full["split"] == "test"].copy()

    # 获取候选特征列
    candidates = [
        c for c in _train.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(_train[c])
    ]

    if mode == "raw_all":
        return candidates, _train, _test

    # top_k / ts_topk：按训练集相关性排序
    y = _train[target_col]
    scores = []
    for col in candidates:
        pearson = _safe_corr(_train[col], y)
        spearman = _safe_corr(_train[col].rank(), y.rank())
        score = float(np.mean([abs(pearson), abs(spearman)]))
        scores.append((col, score))

    scores.sort(key=lambda x: -x[1])
    selected = [s[0] for s in scores[:k]]

    # ts_topk 模式附加信息打印
    if mode == "ts_topk":
        ts_names = set(_get_ts_feature_names())
        ts_selected = [f for f in selected if f in ts_names]
        raw_selected = [f for f in selected if f not in ts_names]
        print(f"  特征选择 ts_topk: {len(selected)} 个特征 "
              f"(含 {len(ts_selected)} 个时序特征, {len(raw_selected)} 个其他)")

    return selected, _train, _test
