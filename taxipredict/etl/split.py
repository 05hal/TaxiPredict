"""训练 / 测试 / 验证集划分。"""

from __future__ import annotations

import pandas as pd


def split_june_last_week(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以 6 月最后 7 天为测试集，其余为训练集。"""
    out = df.copy()

    if "datetime" not in out.columns:
        raise ValueError("数据需要 datetime 列")

    if not pd.api.types.is_datetime64_any_dtype(out["datetime"]):
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")

    june = out[out["datetime"].dt.month == 6].copy()
    if june.empty:
        raise ValueError("数据中没有 6 月，无法按 june_last_week 划分")

    max_day = june["datetime"].dt.normalize().max()
    test_start = max_day - pd.Timedelta(days=6)

    train = out[out["datetime"] < test_start].copy()
    test = out[
        (out["datetime"] >= test_start) & (out["datetime"].dt.month == 6)
    ].copy()

    print(f"  划分: train={len(train)} 行, test={len(test)} 行 (test_start={test_start})")
    return train, test


def split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序比例划分训练 / 验证 / 测试。"""
    out = df.sort_values("datetime").reset_index(drop=True)
    n = len(out)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = out.iloc[:train_end].copy()
    val = out.iloc[train_end:val_end].copy()
    test = out.iloc[val_end:].copy()

    print(f"  划分: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
