"""密集特征构建（天气分箱、滚动统计、滞后特征、交互特征、常量列删除）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


_NEVER_DROP_COLS = {
    "pickups", "geohash", "datetime", "latitude", "longitude",
    "year", "month", "day", "hour", "time_cat", "day_cat",
    "split", "is_test", "source_file",
}


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """删除所有值完全相同的列（排除 _NEVER_DROP_COLS）。"""
    out = df.copy()
    drops = []
    for col in out.columns:
        if col in _NEVER_DROP_COLS:
            continue
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and out[col].nunique(dropna=False) <= 1:
            drops.append(col)
    if drops:
        out = out.drop(columns=drops)
    return out


def build_dense_features(
    df: pd.DataFrame,
    feature_cfg: dict | None = None,
) -> pd.DataFrame:
    """构建稠密特征 DataFrame。

    功能：
    1. 天气分箱：precip_level, temp_bin, low_vis, high_wind
    2. 交互特征：peak_x_precip, weekend_x_precip, evening_x_low_vis 等
    3. 天气滚动统计：precip_3h_sum, temp_3h_delta 等
    4. 客流滞后特征：pickups_lag1, pickups_lag48, lag336 等
    5. 客流滚动均值：pickups_roll6_mean, expanding_mean 等
    6. 对数变换：pickups_lag1_log1p, y_log1p
    7. 差分特征：pickups_mom1, pickups_diff_24h
    8. 缺失标记
    9. 自动删除常量列
    """
    if feature_cfg is None:
        feature_cfg = {}

    out = df.copy()

    # ── 1. 确保 key 列存在 ──
    for col in ["geohash", "datetime", "pickups"]:
        if col not in out.columns:
            raise ValueError(f"输入数据缺少必要列: {col}")

    # ── 2. 天气分箱 ──
    for col in ["precip", "temp", "wspd", "wind_speed", "vis", "visibility", "rhum", "humidity"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    precip_col = "precip" if "precip" in out.columns else None
    if precip_col is not None:
        out["precip_level"] = pd.cut(
            out[precip_col].fillna(0),
            bins=[-1e-9, 0.0, 0.2, 1.0, 5.0, float("inf")],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype("Int64").fillna(0).astype(int)

    temp_col = next((c for c in ["temp", "temperature"] if c in out.columns), None)
    if temp_col is not None:
        out["temp_bin"] = pd.cut(
            out[temp_col],
            bins=[-float("inf"), 0.0, 10.0, 20.0, 30.0, float("inf")],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype("Int64").fillna(2).astype(int)

    vis_col = next((c for c in ["vis", "visibility"] if c in out.columns), None)
    wspd_col = next((c for c in ["wspd", "wind_speed"] if c in out.columns), None)

    if vis_col is not None:
        out["low_vis"] = (out[vis_col] < 8.0).fillna(False).astype(int)
    if wspd_col is not None:
        out["high_wind"] = (out[wspd_col] > 8.0).fillna(False).astype(int)

    # ── 3. 交互特征 ──
    is_peak = out.get("is_peak", out.get("is_morning_peak", pd.Series(0, index=out.index)))
    if "is_evening_peak" in out.columns:
        is_peak = is_peak | out["is_evening_peak"].astype(bool)
    is_peak = pd.to_numeric(is_peak, errors="coerce").fillna(0).astype(int)

    weekend_col = out.get("weekend", pd.Series(0, index=out.index))
    weekend = pd.to_numeric(weekend_col, errors="coerce").fillna(0).astype(int)

    if "is_precip" in out.columns:
        is_precip = pd.to_numeric(out["is_precip"], errors="coerce").fillna(0).astype(int)
    else:
        is_precip = (out.get("precip_level", pd.Series(0)) > 0).astype(int) if "precip_level" in out.columns else pd.Series(0, index=out.index)

    out["peak_x_precip"] = (is_peak * is_precip).astype(int)
    out["weekend_x_precip"] = (weekend * is_precip).astype(int)

    # evening_x_low_vis（旧代码中存在的交互）
    is_evening = pd.to_numeric(out.get("is_evening_peak", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    low_vis_vals = out.get("low_vis", pd.Series(0, index=out.index))
    out["evening_x_low_vis"] = (is_evening * low_vis_vals).astype(int)

    # 缺失标记
    for col in [temp_col, vis_col, wspd_col, precip_col]:
        if col and col in out.columns:
            out[f"missing_{col}"] = out[col].isna().astype(int)

    # ── 4. 天气滚动统计（全局时间聚合） ──
    weather_cols = [c for c in ["temp", "wspd", "wind_speed", "vis", "visibility", "precip", "is_precip"]
                    if c in out.columns]
    if weather_cols and "datetime" in out.columns:
        w = out[["datetime"] + weather_cols].copy()
        w = w.groupby("datetime", as_index=False).mean(numeric_only=True).sort_values("datetime").reset_index(drop=True)
        for c in ["precip", "is_precip", "temp", "vis", "visibility", "wspd", "wind_speed"]:
            if c not in w.columns:
                w[c] = 0.0 if c in ("precip", "is_precip") else np.nan

        roll_hours = feature_cfg.get("weather_roll_hours", [3, 6])
        for rh in roll_hours:
            steps = rh * 2
            precip_key = "precip" if "precip" in w.columns else None
            if precip_key:
                w[f"precip_{rh}h_sum"] = w[precip_key].fillna(0).rolling(steps, min_periods=1).sum()
            is_p_key = "is_precip" if "is_precip" in w.columns else None
            if is_p_key:
                w[f"is_precip_{rh}h_any"] = (w[is_p_key].fillna(0).rolling(steps, min_periods=1).max() > 0).astype(int)

        if "temp" in w.columns:
            w["temp_3h_delta"] = w["temp"] - w["temp"].shift(6)
        if vis_col and vis_col in w.columns:
            w["vis_3h_min"] = w[vis_col].rolling(6, min_periods=1).min()
        if wspd_col and wspd_col in w.columns:
            w["wspd_3h_max"] = w[wspd_col].rolling(6, min_periods=1).max()

        roll_keep = ["datetime"] + [c for c in w.columns if c not in out.columns and c != "datetime" and c not in weather_cols]
        out = out.merge(w[roll_keep], on="datetime", how="left")

    # ── 5. 客流时序特征 ──
    if "geohash" in out.columns and "datetime" in out.columns:
        out = out.sort_values(["geohash", "datetime"]).reset_index(drop=True)
        out["pickups"] = pd.to_numeric(out["pickups"], errors="coerce").fillna(0).astype(int)

        lag_hours = feature_cfg.get("lag_hours", [1, 2, 3, 6, 24, 48, 336])
        g = out.groupby("geohash", sort=False)["pickups"]
        for lag in lag_hours:
            out[f"pickups_lag{lag}"] = g.shift(lag)

        # 滚动均值
        roll_windows = feature_cfg.get("rolling_windows", [6, 12, 48])
        s = out.get("pickups_lag1", g.shift(1))
        for wsize in roll_windows:
            out[f"pickups_roll{wsize}_mean"] = (
                s.groupby(out["geohash"]).rolling(wsize, min_periods=1).mean().reset_index(level=0, drop=True)
            )

        # expanding_mean（旧代码中存在的特征）
        out["pickups_expanding_mean"] = (
            s.groupby(out["geohash"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        )

        out["missing_pickups_lag1"] = out["pickups_lag1"].isna().astype(int)

        # log1p 变换（旧代码中存在的特征）
        if "pickups_lag1" in out.columns:
            out["pickups_lag1_log1p"] = np.log1p(out["pickups_lag1"].fillna(0).clip(lower=0))
        if "pickups_lag48" in out.columns:
            out["pickups_lag48_log1p"] = np.log1p(out["pickups_lag48"].fillna(0).clip(lower=0))

        # 一阶动量 mom1（旧代码中存在的特征）
        if "pickups_lag1" in out.columns and "pickups_lag2" in out.columns:
            out["pickups_mom1"] = out["pickups_lag1"] - out["pickups_lag2"]

        if "pickups_lag1" in out.columns and "pickups_lag48" in out.columns:
            out["pickups_diff_24h"] = out["pickups_lag1"] - out["pickups_lag48"]

        if "pickups_roll48_mean" in out.columns and out["pickups_roll48_mean"].notna().any():
            out["pickups_ratio_roll48"] = out["pickups_lag1"] / (out["pickups_roll48_mean"] + 1e-6)

        # log1p 目标
        out["y_log1p"] = np.log1p(out["pickups"].astype(float))

        # ── 追加：报告中的高级特征 ──

        # weather_severity: 降水/低能见度/强风加权综合 (0~1)
        sev = 0.0
        if "precip_level" in out.columns:
            sev += out["precip_level"].astype(float) / 4.0
        if "low_vis" in out.columns:
            sev += out["low_vis"].astype(float)
        if "high_wind" in out.columns:
            sev += out["high_wind"].astype(float)
        out["weather_severity"] = (sev / 3.0).clip(0, 1)

        # temp_change_1h: 30 分钟粒度下 shift(2)=1 小时
        if "temp" in out.columns:
            out["temp_change_1h"] = out.groupby("geohash")["temp"].diff(2)

        # 分位数分箱（常量列回退到中位数分箱=全 0）
        for src_col, out_col in [("vis", "vis_level"), ("wspd", "wind_level")]:
            if src_col in out.columns:
                try:
                    if out[src_col].nunique() >= 4:
                        out[out_col] = (
                            pd.qcut(out[src_col], q=4, labels=False, duplicates="drop")
                            .astype(float).fillna(0)
                        )
                    else:
                        out[out_col] = pd.cut(out[src_col], bins=4, labels=False).astype(float).fillna(0)
                except Exception:
                    out[out_col] = 0
                out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(0).astype(int)

        # geo_prefix: geohash 前 5 位
        if "geohash" in out.columns:
            out["geo_prefix"] = out["geohash"].astype(str).str[:5]

        # 交互特征
        i_peak = pd.to_numeric(out.get("is_peak", 0), errors="coerce").fillna(0)
        i_weekend = pd.to_numeric(out.get("weekend", 0), errors="coerce").fillna(0)
        i_holiday = pd.to_numeric(out.get("is_holiday", 0), errors="coerce").fillna(0)
        i_precip = pd.to_numeric(out.get("is_precip", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
        temp_norm = pd.to_numeric(out.get("temp", 25), errors="coerce").fillna(25) / 50.0
        rhum_norm = pd.to_numeric(out.get("rhum", out.get("humidity", 50)), errors="coerce").fillna(50) / 100.0
        wspd_val = pd.to_numeric(out.get("wspd", out.get("wind_speed", 5)), errors="coerce").fillna(5) / 20.0

        out["peak_weekend"] = (i_peak * i_weekend).astype(int)
        out["holiday_peak"] = (i_holiday * i_peak).astype(int)
        out["rain_peak"] = (i_precip * i_peak).astype(int)
        out["temp_rhum"] = (temp_norm * rhum_norm).astype(float)
        out["temp_wspd"] = (temp_norm * wspd_val).astype(float)

    # ── 6. 删除常量列 ──
    if feature_cfg.get("drop_constant", True):
        out = drop_constant_columns(out)

    return out
