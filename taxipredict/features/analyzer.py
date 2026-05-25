"""特征相关性 + 熵值分析。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """安全的相关系数计算，处理 NaN / Inf / 常量。"""
    temp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(temp) < 2 or temp["x"].nunique() <= 1 or temp["y"].nunique() <= 1:
        return 0.0
    value = temp["x"].corr(temp["y"], method=method)
    return 0.0 if pd.isna(value) else float(value)


def calc_entropy_for_series(s: pd.Series, bins: int = 20) -> tuple[float, float]:
    """计算数值序列的信息熵和归一化熵。"""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return 0.0, 0.0
    try:
        hist, _ = np.histogram(s, bins=bins)
    except Exception:
        return 0.0, 0.0
    if hist.sum() == 0:
        return 0.0, 0.0
    prob = hist / hist.sum()
    ent = float(entropy(prob, base=2))
    norm = ent / np.log2(bins) if bins > 1 else 0.0
    return ent, norm


def generate_report(
    df: pd.DataFrame,
    target_col: str = "pickups",
) -> pd.DataFrame:
    """生成特征分析报告。

    输出字段：
    - feature: 特征名
    - pearson_corr: Pearson 相关系数
    - spearman_corr: Spearman 相关系数
    - importance_score: (|pearson| + |spearman|) / 2
    - entropy: 信息熵
    - normalized_entropy: 归一化熵
    - unique_count: 唯一值数
    - missing_rate: 缺失率
    """
    rows = []
    exclude_cols = {target_col, "datetime", "geohash", "latitude", "longitude",
                    "time_cat", "day_cat", "split", "source_file", "is_test"}

    y = df[target_col]

    for col in df.columns:
        if col in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = df[col]
        missing_rate = s.isna().mean()
        n_unique = int(s.nunique())

        pearson = safe_corr(s, y, "pearson")
        spearman = safe_corr(s, y, "spearman")
        importance = float(np.mean([abs(pearson), abs(spearman)]))

        ent, norm_ent = calc_entropy_for_series(s)

        rows.append({
            "feature": col,
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "importance_score": importance,
            "entropy": ent,
            "normalized_entropy": norm_ent,
            "unique_count": n_unique,
            "missing_rate": missing_rate,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
    return result
