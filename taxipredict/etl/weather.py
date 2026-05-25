"""天气数据加载与融合。"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_weather(weather_path: str | Path, freq: str = "1h") -> pd.DataFrame:
    """加载 NOAA 天气 CSV，按小时对齐。

    自动检测字段命名格式（NOAA 标准 / 简写）。
    """
    from taxipredict.etl.loader import safe_read_csv

    w = safe_read_csv(weather_path)

    if "DATE" not in w.columns and "date" not in w.columns:
        raise ValueError("天气 CSV 必须包含 DATE 或 date 列")

    date_col = "DATE" if "DATE" in w.columns else "date"
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w = w[w[date_col].notna()].copy()

    w["time_bin"] = w[date_col].dt.floor(freq)
    w = w.sort_values(["time_bin", date_col]).groupby("time_bin", as_index=False).tail(1)

    # 统一列名映射
    column_map = {
        "HourlyDryBulbTemperature": "temp",
        "HourlyRelativeHumidity": "humidity",
        "HourlyWindSpeed": "wind_speed",
        "HourlyVisibility": "visibility",
        "HourlyPrecipitation": "precip",
        "HourlyStationPressure": "pressure",
        "HourlyDewPointTemperature": "dew_point",
        "TMAX": "temp_max",
        "TMIN": "temp_min",
        "PRCP": "precip_daily",
        "AWND": "avg_wind",
        "HourlyPresentWeather": "present_weather",
    }
    for src, dst in column_map.items():
        if src in w.columns and dst not in w.columns:
            w[dst] = pd.to_numeric(w[src], errors="coerce")

    # 确保必要列存在
    required_cols = ["temp", "humidity", "wind_speed", "visibility", "precip"]
    for col in required_cols:
        if col not in w.columns:
            w[col] = np.nan

    # 派生的布尔特征
    w["is_precip"] = (w["precip"].fillna(0) > 0).astype(float)
    w["low_visibility"] = (w["visibility"] < 5).fillna(False).astype(float)
    w["high_wind"] = (w["wind_speed"] > 10).fillna(False).astype(float)

    weather_cols = ["time_bin", "temp", "humidity", "wind_speed", "visibility", "precip",
                    "is_precip", "low_visibility", "high_wind"]
    extra = [c for c in ["pressure", "dew_point", "temp_max", "temp_min",
                         "precip_daily", "avg_wind", "present_weather"]
             if c in w.columns]
    weather_cols.extend(extra)

    result = w[weather_cols].copy()
    print(f"  天气数据: {len(result)} 行, {result['time_bin'].min()} ~ {result['time_bin'].max()}")
    return result


def join_weather(taxi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """左连接天气到客流量数据（按时间戳对齐），同时添加假日/高峰/气象编码。"""
    out = taxi_df.copy()

    if "time_bin" not in out.columns and "datetime" in out.columns:
        out["time_bin"] = out["datetime"].dt.floor("1h")

    weather_join = weather_df.rename(columns=lambda c: f"weather_{c}" if c != "time_bin" else c)
    out = out.merge(weather_join, on="time_bin", how="left")

    # 填充缺失天气
    for col in out.columns:
        if col.startswith("weather_") and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0)

    out = out.drop(columns=["time_bin"])

    # 追加假日标记
    if "datetime" in out.columns:
        out = add_holiday_flag(out)
        out = add_peak_flags(out)

    # 解析气象编码（如果存在）
    weather_code_col = next((c for c in out.columns if "present_weather" in c), None)
    if weather_code_col is not None:
        weather_flags = parse_present_weather(out[weather_code_col])
        for col in weather_flags.columns:
            out[col] = weather_flags[col]
        out = out.drop(columns=[weather_code_col])

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加三角函数时间特征。"""
    out = df.copy()

    hour = out["datetime"].dt.hour
    minute = out["datetime"].dt.minute
    time_num = (hour * 60 + minute + 30.0) / (24.0 * 60.0)
    day_of_week = out["datetime"].dt.dayofweek

    out["time_num"] = time_num
    out["time_cos"] = np.cos(time_num * 2 * math.pi)
    out["time_sin"] = np.sin(time_num * 2 * math.pi)
    out["day_num"] = (day_of_week + time_num) / 7.0
    out["day_cos"] = np.cos(out["day_num"] * 2 * math.pi)
    out["day_sin"] = np.sin(out["day_num"] * 2 * math.pi)
    out["weekend"] = (day_of_week >= 5).astype(int)
    out["hour"] = hour
    out["time_cat"] = hour.map(lambda h: f"{int(h):02d}:00")

    return out


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """标记美国联邦假日（is_holiday）。"""
    out = df.copy()
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        dts = out["datetime"].dt.normalize()
        holidays = cal.holidays(start=dts.min(), end=dts.max()).normalize()
        out["is_holiday"] = dts.isin(holidays).astype(int)
    except Exception:
        out["is_holiday"] = 0
    return out


def add_peak_flags(df: pd.DataFrame) -> pd.DataFrame:
    """标记早晚高峰。"""
    out = df.copy()
    hour = pd.to_numeric(out.get("hour", out["datetime"].dt.hour), errors="coerce").fillna(-1)
    morning = hour.between(7, 9, inclusive="both")
    evening = hour.between(16, 19, inclusive="both")
    out["is_morning_peak"] = morning.fillna(False).astype(int)
    out["is_evening_peak"] = evening.fillna(False).astype(int)
    out["is_peak"] = ((out["is_morning_peak"] == 1) | (out["is_evening_peak"] == 1)).astype(int)
    return out


def parse_present_weather(series: pd.Series) -> pd.DataFrame:
    """解析 NOAA HourlyPresentWeather 编码，返回雨/雪/雾标记。"""
    raw = series.fillna("").astype(str).str.strip()

    def _normalize(val: str) -> list[str]:
        if not val or val.lower() in {"nan", "<na>", "none", "null", "0"}:
            return []
        tokens = []
        for part in val.split("|"):
            code = part.split(":", 1)[0].strip()
            code = re.sub(r"^[+-]", "", code)
            code = re.sub(r"[^A-Z]", "", code.upper())
            if code:
                tokens.append(code)
        return tokens

    def _has_any(toks: list[str], predicate) -> int:
        return int(any(predicate(t) for t in toks))

    normed = raw.map(_normalize)
    is_rain = normed.map(lambda t: _has_any(t, lambda c: "RA" in c or c.endswith("RA"))).astype(int)
    is_snow = normed.map(lambda t: _has_any(t, lambda c: "SN" in c)).astype(int)
    is_fog = normed.map(lambda t: _has_any(t, lambda c: "FG" in c)).astype(int)
    return pd.DataFrame({"is_rain": is_rain, "is_snow": is_snow, "is_fog": is_fog})
