from __future__ import annotations
import argparse
import math
import re
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar




DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SCRIPT_DIR = Path(__file__).resolve().parent


def time_features_for_hour(hour: pd.Series) -> pd.DataFrame:
    time_num = (hour.astype(float) + 0.5) / 24.0
    time_cos = (time_num * 2 * math.pi).map(math.cos)
    time_sin = (time_num * 2 * math.pi).map(math.sin)
    time_cat = hour.astype(int).map(lambda h: f"{h:02d}:00")
    return pd.DataFrame(
        {"time_cat": time_cat, "time_num": time_num, "time_cos": time_cos, "time_sin": time_sin}
    )


def day_features(dt_hour: pd.Series) -> pd.DataFrame:
    weekday = dt_hour.dt.weekday.astype(int)
    time_num = (dt_hour.dt.hour.astype(float) + 0.5) / 24.0
    day_num = (weekday.astype(float) + time_num) / 7.0
    day_cos = (day_num * 2 * math.pi).map(math.cos)
    day_sin = (day_num * 2 * math.pi).map(math.sin)
    weekend = weekday.isin([5, 6]).astype(int)
    day_cat = weekday.map(lambda w: DAY_NAMES[w])
    return pd.DataFrame(
        {
            "day_cat": day_cat,
            "day_num": day_num,
            "day_cos": day_cos,
            "day_sin": day_sin,
            "weekend": weekend,
        }
    )


def normalize_present_weather(value: str) -> list[str]:
    if value is None or pd.isna(value):
        return []

    value = str(value).strip()
    if not value or value.lower() in {"nan", "<na>", "none", "null"}:
        return []

    parts = [p.strip() for p in value.split("|")]
    tokens: list[str] = []
    for p in parts:
        if not p:
            continue
        code = p.split(":", 1)[0].strip()
        code = re.sub(r"^[+-]", "", code)
        code = re.sub(r"[^A-Z]", "", code.upper())
        if code:
            tokens.append(code)
    return tokens


def build_weather_flags(present_weather: pd.Series) -> pd.DataFrame:
    tokens_series = present_weather.map(normalize_present_weather)

    def has_any(tokens: list[str], predicate) -> int:
        return int(any(predicate(t) for t in tokens))

    is_rain = tokens_series.map(lambda toks: has_any(toks, lambda t: "RA" in t or t.endswith("RA")))
    is_snow = tokens_series.map(lambda toks: has_any(toks, lambda t: "SN" in t))
    is_fog = tokens_series.map(lambda toks: has_any(toks, lambda t: "FG" in t))
    return pd.DataFrame({"is_rain": is_rain, "is_snow": is_snow, "is_fog": is_fog})


def parse_precip(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = series.astype(str).str.strip()
    is_trace = raw.eq("T")
    precip = pd.to_numeric(raw.where(~is_trace, "0"), errors="coerce")
    is_precip = ((precip.fillna(0) > 0) | is_trace).astype(int)
    return precip, is_precip


def quantile_level(series: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=numeric.index, dtype="Int64")
    valid = numeric.dropna()
    unique_count = valid.nunique()
    if unique_count == 0:
        return result
    if unique_count == 1:
        result.loc[valid.index] = 0
        return result

    q = min(bins, unique_count)
    ranked = valid.rank(method="first")
    binned = pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype("Int64")
    result.loc[binned.index] = binned
    return result


def add_dense_weather_features(weather_hourly: pd.DataFrame) -> pd.DataFrame:
    w = weather_hourly.sort_values("dt_hour").copy()

    precip = w["precip"].fillna(0)
    vis = w["vis"]
    wspd = w["wspd"]
    temp = w["temp"]

    w["precip_level"] = pd.cut(
        precip,
        bins=[-0.001, 0, 0.01, 0.1, 0.3, float("inf")],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype("Int64")
    w["vis_level"] = pd.cut(
        vis,
        bins=[-float("inf"), 4, 10, 15, float("inf")],
        labels=[3, 2, 1, 0],
    ).astype("Int64")
    w["wind_level"] = quantile_level(wspd, bins=4)
    w["temp_level"] = quantile_level(temp, bins=5)

    w["weather_severity"] = (
        w["precip_level"].fillna(0).astype(float)
        + w["vis_level"].fillna(0).astype(float)
        + w["wind_level"].fillna(0).astype(float)
        + w["is_rain"].fillna(0).astype(float)
        + w["is_fog"].fillna(0).astype(float)
        + ((temp < 0) | (temp > 27)).fillna(False).astype(float)
    )

    w["precip_lag_1h"] = precip.shift(1)
    w["precip_rolling_3h"] = precip.rolling(3, min_periods=1).sum()
    w["temp_change_1h"] = temp.diff()
    w["vis_change_1h"] = vis.diff()
    w["weather_severity_lag_1h"] = w["weather_severity"].shift(1)
    w["weather_severity_rolling_3h"] = (
        w["weather_severity"].rolling(3, min_periods=1).mean()
    )
    w["bad_weather"] = (w["weather_severity"] >= 2).astype(int)

    return w


def load_weather_hourly(weather_csv: Path) -> pd.DataFrame:
    w = pd.read_csv(weather_csv, low_memory=False)

    if "DATE" not in w.columns or "REPORT_TYPE" not in w.columns:
        raise ValueError("Weather CSV must contain DATE and REPORT_TYPE columns")

    w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
    w = w[w["DATE"].notna()].copy()

    w = w[w["REPORT_TYPE"].isin(["FM-15", "FM-16"])].copy()
    w["dt_hour"] = w["DATE"].dt.floor("h")

    w = w.sort_values(["dt_hour", "DATE"])
    w = w.groupby("dt_hour", as_index=False).tail(1)

    keep = [
        "dt_hour",
        "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity",
        "HourlyWindSpeed",
        "HourlyVisibility",
        "HourlyPrecipitation",
        "HourlyPresentWeatherType",
    ]
    for col in keep:
        if col not in w.columns:
            w[col] = pd.NA
    w = w[keep].copy()

    w = w.rename(
        columns={
            "HourlyDryBulbTemperature": "temp",
            "HourlyRelativeHumidity": "rhum",
            "HourlyWindSpeed": "wspd",
            "HourlyVisibility": "vis",
            "HourlyPrecipitation": "precip",
            "HourlyPresentWeatherType": "present_weather",
        }
    )

    w["temp"] = pd.to_numeric(w["temp"], errors="coerce")
    w["rhum"] = pd.to_numeric(w["rhum"], errors="coerce")
    w["wspd"] = pd.to_numeric(w["wspd"], errors="coerce")
    w["vis"] = pd.to_numeric(w["vis"], errors="coerce")
    w["precip"], w["is_precip"] = parse_precip(w["precip"])
    w = pd.concat([w, build_weather_flags(w["present_weather"])], axis=1)
    w = w.drop(columns=["present_weather"])
    w = add_dense_weather_features(w)
    return w


def estimate_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("rb") as fp:
            line_count = 0
            while True:
                chunk = fp.read(1024 * 1024)
                if not chunk:
                    break
                line_count += chunk.count(b"\n")
        return max(0, line_count - 1)
    except OSError:
        return None


def build_us_federal_holiday_index(years: set[int]) -> pd.DatetimeIndex:
    if not years:
        return pd.DatetimeIndex([])

    cal = USFederalHolidayCalendar()
    start = f"{min(years)}-01-01"
    end = f"{max(years)}-12-31"
    return cal.holidays(start=start, end=end).normalize()


def add_peak_flags(
    hour: pd.Series,
    morning_start: int,
    morning_end: int,
    evening_start: int,
    evening_end: int,
) -> pd.DataFrame:
    h = pd.to_numeric(hour, errors="coerce")
    morning = h.between(morning_start, morning_end, inclusive="both")
    evening = h.between(evening_start, evening_end, inclusive="both")
    is_morning_peak = morning.fillna(False).astype(int)
    is_evening_peak = evening.fillna(False).astype(int)
    is_peak = ((is_morning_peak == 1) | (is_evening_peak == 1)).astype(int)
    return pd.DataFrame(
        {
            "is_morning_peak": is_morning_peak,
            "is_evening_peak": is_evening_peak,
            "is_peak": is_peak,
        }
    )


def augment_taxi_chunk(
    taxi_chunk: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    holiday_dates: pd.DatetimeIndex,
    morning_peak_start: int,
    morning_peak_end: int,
    evening_peak_start: int,
    evening_peak_end: int,
) -> pd.DataFrame:
    required = {"year", "month", "day", "time_cat"}
    missing = required - set(taxi_chunk.columns)
    if missing:
        raise ValueError(f"Taxi CSV missing required columns: {sorted(missing)}")

    chunk = taxi_chunk.copy()
    orig_cols = list(chunk.columns)

    year_num = pd.to_numeric(chunk["year"], errors="coerce")
    month_num = pd.to_numeric(chunk["month"], errors="coerce")
    day_num = pd.to_numeric(chunk["day"], errors="coerce")
    hour_num = pd.to_numeric(chunk["time_cat"].astype(str).str.slice(0, 2), errors="coerce")

    dt = pd.to_datetime(
        dict(year=year_num, month=month_num, day=day_num, hour=hour_num),
        errors="coerce",
    )
    chunk["dt_hour"] = dt.dt.floor("h")

    merged = chunk.merge(weather_hourly, on="dt_hour", how="left")
    merged = merged.reset_index(drop=True)

    day = merged["dt_hour"].dt.normalize()
    merged["is_holiday"] = day.isin(holiday_dates).fillna(False).astype(int)

    peaks = add_peak_flags(
        hour_num,
        morning_peak_start,
        morning_peak_end,
        evening_peak_start,
        evening_peak_end,
    ).reset_index(drop=True)
    merged = pd.concat([merged, peaks], axis=1)

    merged = merged.drop(columns=["dt_hour"])

    new_cols = [
        "temp",
        "rhum",
        "wspd",
        "vis",
        "precip",
        "is_precip",
        "is_rain",
        "is_snow",
        "is_fog",
        "precip_level",
        "vis_level",
        "wind_level",
        "temp_level",
        "weather_severity",
        "precip_lag_1h",
        "precip_rolling_3h",
        "temp_change_1h",
        "vis_change_1h",
        "weather_severity_lag_1h",
        "weather_severity_rolling_3h",
        "bad_weather",
        "is_holiday",
        "is_morning_peak",
        "is_evening_peak",
        "is_peak",
    ]
    existing_new = [c for c in new_cols if c in merged.columns and c not in orig_cols]
    merged = merged[orig_cols + existing_new]

    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxi-csv",
        type=Path,
        default=SCRIPT_DIR / "jun14" / "taxi_prediction_style_aggregated.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=SCRIPT_DIR / "LCD_USW00094728_2014.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=SCRIPT_DIR / "jun14" / "taxi_prediction_hourly_with_weather.csv",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--progress-every", type=int, default=200_000)
    parser.add_argument("--morning-peak-start", type=int, default=7)
    parser.add_argument("--morning-peak-end", type=int, default=9)
    parser.add_argument("--evening-peak-start", type=int, default=16)
    parser.add_argument("--evening-peak-end", type=int, default=19)

    args = parser.parse_args()

    if args.out_csv.resolve() == args.taxi_csv.resolve():
        raise ValueError("out-csv must be different from taxi-csv (refusing to overwrite input)")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Loading hourly weather...")
    weather_hourly = load_weather_hourly(args.weather_csv)

    print("Scanning taxi CSV for years and total rows...")
    taxi_head = pd.read_csv(args.taxi_csv, nrows=50, dtype=str, keep_default_na=False)
    years = set(pd.to_numeric(taxi_head.get("year"), errors="coerce").dropna().astype(int).tolist())
    if not years:
        years = {2014}

    total_rows = estimate_csv_rows(args.taxi_csv)
    holiday_dates = build_us_federal_holiday_index(years)

    if total_rows is not None:
        print(f"Taxi rows (estimated): {total_rows:,}")
    print(f"Holiday years: {sorted(years)}")

    processed = 0
    wrote_header = False

    print("Augmenting taxi CSV with weather + holiday + peak-hour features...")
    for chunk in pd.read_csv(args.taxi_csv, chunksize=args.chunksize, dtype=str, keep_default_na=False):
        out_chunk = augment_taxi_chunk(
            chunk,
            weather_hourly,
            holiday_dates,
            args.morning_peak_start,
            args.morning_peak_end,
            args.evening_peak_start,
            args.evening_peak_end,
        )

        out_chunk.to_csv(
            args.out_csv,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
            encoding="utf-8",
        )
        wrote_header = True

        processed += len(chunk)
        if args.progress_every > 0 and processed % args.progress_every < len(chunk):
            if total_rows:
                pct = processed / float(total_rows) * 100.0
                print(f"Processed: {processed:,}/{total_rows:,} ({pct:.1f}%)")
            else:
                print(f"Processed: {processed:,}")

    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {processed:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())