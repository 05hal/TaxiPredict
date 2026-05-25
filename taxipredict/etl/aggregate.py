"""空间聚合：将原始订单按 geohash + 时间桶聚合。"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

REGION_GROUP_COL = "geohash"


def aggregate_to_hourly(
    df: pd.DataFrame,
    target_col: str = "pickups",
) -> pd.DataFrame:
    """按 geohash + 日期 + 整点小时 聚合需求量。

    聚合规则：
    - pickups（目标列）: sum
    - 数值特征: mean
    - 类别/字符串: first
    """
    out = df.copy()

    # 确保 datetime 列存在
    if "datetime" not in out.columns:
        from taxipredict.etl.loader import build_datetime
        out = build_datetime(out)

    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["hour"] = out["datetime"].dt.hour

    group_cols = [REGION_GROUP_COL, "year", "month", "day", "hour"]

    # 构造聚合规格
    skip_agg = {"datetime", target_col, REGION_GROUP_COL, "year", "month", "day", "hour"}
    agg_spec: Dict[str, str] = {target_col: "sum"}

    for col in out.columns:
        if col in skip_agg or col in agg_spec:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            agg_spec[col] = "mean"
        else:
            agg_spec[col] = "first"

    before = len(out)
    out = out.groupby(group_cols, as_index=False).agg(agg_spec)

    # 重新构造 datetime
    out["datetime"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=out["day"], hour=out["hour"])
    )

    out = out.sort_values([REGION_GROUP_COL, "datetime"]).reset_index(drop=True)

    region_count = out[REGION_GROUP_COL].nunique()
    slot_count = len(out)
    print(f"  聚合: {before:,} 行 → {slot_count:,} 行 ({region_count} 个区域)")

    return out


def aggregate_by_grid(
    df: pd.DataFrame,
    grid_size: float = 0.02,
    target_col: str = "pickups",
) -> pd.DataFrame:
    """按经纬度网格 + 时间聚合（STID 用）。"""
    out = df.copy()

    out["lat_grid"] = np.floor(out["Lat"].astype(float) / grid_size).astype("Int64")
    out["lon_grid"] = np.floor(out["Lon"].astype(float) / grid_size).astype("Int64")
    out["region_id"] = out["lat_grid"].astype(str) + "_" + out["lon_grid"].astype(str)

    if "datetime" not in out.columns:
        from taxipredict.etl.loader import build_datetime
        out = build_datetime(out)

    out["time_bin"] = out["datetime"].dt.floor("1h")
    out["year"] = out["datetime"].dt.year
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["hour"] = out["datetime"].dt.hour

    group_cols = ["region_id", "time_bin", "year", "month", "day", "hour"]
    skip_agg = {"datetime", "time_bin", target_col, "region_id", "year", "month", "day", "hour",
                "Lat", "Lon", "lat_grid", "lon_grid", "Date/Time"}

    agg_spec: Dict[str, str] = {target_col: "sum"}
    for col in out.columns:
        if col in skip_agg or col in agg_spec:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            agg_spec[col] = "mean"
        else:
            agg_spec[col] = "first"

    out = out.groupby(group_cols, as_index=False).agg(agg_spec)
    out["datetime"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=out["day"], hour=out["hour"])
    )
    out = out.sort_values(["region_id", "datetime"]).reset_index(drop=True)

    print(f"  网格聚合: {len(out)} 行, {out['region_id'].nunique()} 个区域")
    return out
