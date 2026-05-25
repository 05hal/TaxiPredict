import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_dt_from_columns(df: pd.DataFrame) -> pd.Series:
    hour = pd.to_numeric(df["time_cat"].astype(str).str.slice(0, 2), errors="coerce")
    minute = pd.to_numeric(df["time_cat"].astype(str).str.slice(3, 5), errors="coerce")
    dt = pd.to_datetime(
        dict(
            year=pd.to_numeric(df["year"], errors="coerce"),
            month=pd.to_numeric(df["month"], errors="coerce"),
            day=pd.to_numeric(df["day"], errors="coerce"),
            hour=hour,
            minute=minute,
        ),
        errors="coerce",
    )
    return dt


def add_dense_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["temp", "rhum", "wspd", "vis", "precip"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    for col in ["is_precip", "is_holiday", "is_morning_peak", "is_evening_peak", "is_peak"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            out[col] = 0

    precip = out["precip"].fillna(0)
    vis = out["vis"]
    wspd = out["wspd"]
    temp = out["temp"]

    out["precip_level"] = pd.cut(
        precip,
        bins=[-1e-9, 0.0, 0.2, 1.0, 5.0, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype("Int64")
    out["precip_level"] = out["precip_level"].fillna(0).astype(int)

    out["low_vis"] = (vis < 8.0).fillna(False).astype(int)
    out["high_wind"] = (wspd > 8.0).fillna(False).astype(int)

    out["temp_bin"] = pd.cut(
        temp,
        bins=[-float("inf"), 0.0, 10.0, 20.0, 30.0, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype("Int64")
    out["temp_bin"] = out["temp_bin"].fillna(2).astype(int)

    weekend = pd.to_numeric(out.get("weekend", 0), errors="coerce").fillna(0).astype(int)

    out["peak_x_precip"] = (out["is_peak"] * out["is_precip"]).astype(int)
    out["peak_x_precip_level"] = (out["is_peak"] * out["precip_level"]).astype(int)
    out["weekend_x_precip"] = (weekend * out["is_precip"]).astype(int)
    out["evening_x_low_vis"] = (out["is_evening_peak"] * out["low_vis"]).astype(int)

    out["missing_temp"] = out["temp"].isna().astype(int)
    out["missing_vis"] = out["vis"].isna().astype(int)
    out["missing_wspd"] = out["wspd"].isna().astype(int)
    out["missing_precip"] = out["precip"].isna().astype(int)

    return out


def add_weather_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    weather_cols = [c for c in ["temp", "wspd", "vis", "precip", "is_precip"] if c in out.columns]
    w = out[["dt"] + weather_cols].copy()
    w = w.groupby("dt", as_index=False).mean(numeric_only=True).sort_values("dt").reset_index(drop=True)

    if "precip" not in w.columns:
        w["precip"] = 0.0
    if "is_precip" not in w.columns:
        w["is_precip"] = 0.0
    if "temp" not in w.columns:
        w["temp"] = np.nan
    if "vis" not in w.columns:
        w["vis"] = np.nan
    if "wspd" not in w.columns:
        w["wspd"] = np.nan

    w["precip_3h_sum"] = w["precip"].fillna(0).rolling(6, min_periods=1).sum()
    w["precip_6h_sum"] = w["precip"].fillna(0).rolling(12, min_periods=1).sum()
    w["is_precip_6h_any"] = (w["is_precip"].fillna(0).rolling(12, min_periods=1).max() > 0).astype(int)

    w["temp_3h_delta"] = w["temp"] - w["temp"].shift(6)
    w["vis_3h_min"] = w["vis"].rolling(6, min_periods=1).min()
    w["wspd_3h_max"] = w["wspd"].rolling(6, min_periods=1).max()

    out = out.merge(
        w[
            [
                "dt",
                "precip_3h_sum",
                "precip_6h_sum",
                "is_precip_6h_any",
                "temp_3h_delta",
                "vis_3h_min",
                "wspd_3h_max",
            ]
        ],
        on="dt",
        how="left",
    )
    return out


def add_pickup_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["geohash", "dt"]).reset_index(drop=True)

    out["pickups"] = pd.to_numeric(out["pickups"], errors="coerce").fillna(0).astype(int)

    g = out.groupby("geohash", sort=False)["pickups"]
    out["pickups_lag1"] = g.shift(1)
    out["pickups_lag2"] = g.shift(2)
    out["pickups_lag48"] = g.shift(48)
    out["pickups_lag336"] = g.shift(336)

    s = out["pickups_lag1"]
    out["pickups_roll6_mean"] = (
        s.groupby(out["geohash"]).rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["pickups_roll12_mean"] = (
        s.groupby(out["geohash"]).rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["pickups_roll48_mean"] = (
        s.groupby(out["geohash"]).rolling(48, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    out["pickups_expanding_mean"] = (
        s.groupby(out["geohash"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )

    out["missing_pickups_lag1"] = out["pickups_lag1"].isna().astype(int)

    out["pickups_lag1_log1p"] = np.log1p(out["pickups_lag1"])
    out["pickups_lag48_log1p"] = np.log1p(out["pickups_lag48"])
    out["pickups_mom1"] = out["pickups_lag1"] - out["pickups_lag2"]
    out["pickups_diff_24h"] = out["pickups_lag1"] - out["pickups_lag48"]
    out["pickups_ratio_roll48"] = out["pickups_lag1"] / (out["pickups_roll48_mean"] + 1e-6)

    out["y_log1p"] = np.log1p(out["pickups"].astype(float))
    return out


def load_one(path: Path, tag: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    orig_cols = list(df.columns)

    df = df.copy()
    df["_src"] = tag
    df["_row_id"] = range(len(df))
    df["dt"] = parse_dt_from_columns(df)
    df = df[df["dt"].notna()].copy()
    return df, orig_cols


NEVER_DROP_COLS = {
    "year",
    "month",
    "day",
    "time_cat",
    "time_num",
    "time_cos",
    "time_sin",
    "day_cat",
    "day_num",
    "day_cos",
    "day_sin",
    "weekend",
    "geohash",
    "pickups",
    "latitude",
    "longitude",
}


def find_constant_columns(df: pd.DataFrame, candidates: list[str]) -> set[str]:
    drops: set[str] = set()
    for col in candidates:
        if col in NEVER_DROP_COLS or col not in df.columns:
            continue

        s = df[col]
        if s.empty:
            continue

        if s.dtype == object:
            s = s.replace("", pd.NA)

        num = pd.to_numeric(s, errors="coerce")
        check = num if num.notna().any() else s

        if check.dropna().nunique() <= 1:
            drops.add(col)

    return drops


def build_output(df: pd.DataFrame, orig_cols: list[str]) -> pd.DataFrame:
    engineered = [
        "precip_level",
        "low_vis",
        "high_wind",
        "temp_bin",
        "precip_3h_sum",
        "precip_6h_sum",
        "is_precip_6h_any",
        "temp_3h_delta",
        "vis_3h_min",
        "wspd_3h_max",
        "peak_x_precip",
        "peak_x_precip_level",
        "weekend_x_precip",
        "evening_x_low_vis",
        "missing_temp",
        "missing_vis",
        "missing_wspd",
        "missing_precip",
        "pickups_lag1",
        "pickups_lag2",
        "pickups_lag48",
        "pickups_lag336",
        "pickups_roll6_mean",
        "pickups_roll12_mean",
        "pickups_roll48_mean",
        "pickups_expanding_mean",
        "missing_pickups_lag1",
        "pickups_lag1_log1p",
        "pickups_lag48_log1p",
        "pickups_mom1",
        "pickups_diff_24h",
        "pickups_ratio_roll48",
        "y_log1p",
    ]

    keep_orig = [c for c in orig_cols if c in df.columns]
    keep_engineered = [c for c in engineered if c in df.columns and c not in keep_orig]

    candidate_drop = [c for c in (keep_orig + keep_engineered) if c not in NEVER_DROP_COLS]
    drop_cols = find_constant_columns(df, candidate_drop)

    keep_orig = [c for c in keep_orig if c not in drop_cols]
    keep_engineered = [c for c in keep_engineered if c not in drop_cols]

    return df[keep_orig + keep_engineered]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jun-csv",
        type=Path,
        default=Path(
            r"d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\taxi_prediction_hourly_with_weather.csv"
        ),
    )
    parser.add_argument(
        "--may-csv",
        type=Path,
        default=Path(
            r"d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\taxi_prediction_hourly_with_weather.csv"
        ),
    )
    parser.add_argument(
        "--jun-out-csv",
        type=Path,
        default=Path(r"d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\jun14\xgb_dense_features.csv"),
    )
    parser.add_argument(
        "--may-out-csv",
        type=Path,
        default=Path(r"d:\mine\bjtu\交通模型预测\TaxiPredict-dev\new\may14\xgb_dense_features.csv"),
    )
    parser.add_argument("--no-pickup-lags", action="store_true")
    args = parser.parse_args()

    may, may_cols = load_one(args.may_csv, "may14")
    jun, jun_cols = load_one(args.jun_csv, "jun14")

    df = pd.concat([may, jun], ignore_index=True, sort=False)

    df = add_dense_weather_features(df)
    df = add_weather_rolling_features(df)

    if not args.no_pickup_lags:
        df = add_pickup_lag_features(df)
    else:
        df["pickups"] = pd.to_numeric(df["pickups"], errors="coerce").fillna(0).astype(int)
        df["y_log1p"] = np.log1p(df["pickups"].astype(float))

    may_out_df = df[df["_src"] == "may14"].sort_values("_row_id").reset_index(drop=True)
    jun_out_df = df[df["_src"] == "jun14"].sort_values("_row_id").reset_index(drop=True)

    may_out = build_output(may_out_df, may_cols)
    jun_out = build_output(jun_out_df, jun_cols)

    may_out.to_csv(args.may_out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {args.may_out_csv}")
    print(f"Rows: {len(may_out):,}")

    jun_out.to_csv(args.jun_out_csv, index=False, encoding="utf-8")
    print(f"Wrote: {args.jun_out_csv}")
    print(f"Rows: {len(jun_out):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())