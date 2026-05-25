import csv
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geohash

DATASETS = {
    "may14": ROOT / "实验数据" / "uber-raw-data-may14.csv",
    "jun14": ROOT / "实验数据" / "uber-raw-data-jun14.csv",
}
OUTPUT_ROOT = ROOT / "new"
GEOHASH_PRECISION = 7
TIME_BINS_PER_DAY = 48
MINUTES_PER_BIN = int((24 / float(TIME_BINS_PER_DAY)) * 60)
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "pickup_yellow_red",
    ["#fffde7", "#fff176", "#ffb74d", "#fb6a4a", "#d7301f"],
)


def parse_datetime(value):
    return datetime.strptime(value, "%m/%d/%Y %H:%M:%S")


def date_features(dt):
    num_minutes = dt.hour * 60 + dt.minute
    time_bin = num_minutes // MINUTES_PER_BIN
    hour_bin = (time_bin * MINUTES_PER_BIN) // 60
    minute_bin = (time_bin * MINUTES_PER_BIN) % 60
    time_cat = f"{hour_bin:02d}:{minute_bin:02d}"

    time_num = (
        hour_bin * 60 + minute_bin + MINUTES_PER_BIN / 2.0
    ) / (60 * 24)
    time_cos = math.cos(time_num * 2 * math.pi)
    time_sin = math.sin(time_num * 2 * math.pi)

    day_of_week = dt.weekday()
    day_cat = dt.strftime("%A")
    day_num = (day_of_week + time_num) / 7.0
    day_cos = math.cos(day_num * 2 * math.pi)
    day_sin = math.sin(day_num * 2 * math.pi)
    weekend = 1 if day_of_week in (5, 6) else 0

    return (
        dt.year,
        dt.month,
        dt.day,
        time_cat,
        round(time_num, 12),
        time_cos,
        time_sin,
        day_cat,
        day_num,
        day_cos,
        day_sin,
        weekend,
    )


def clean_record(row):
    try:
        dt = parse_datetime(row["Date/Time"])
        lat = float(row["Lat"])
        lon = float(row["Lon"])
        location = geohash.encode(lat, lon, GEOHASH_PRECISION)
    except (KeyError, TypeError, ValueError):
        return None

    return date_features(dt) + (location,)


def aggregate_dataset(input_path):
    grouped = Counter()
    base_counts = Counter()
    raw_points = []
    invalid_rows = 0
    total_rows = 0

    with input_path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            total_rows += 1
            cleaned = clean_record(row)
            if cleaned is None:
                invalid_rows += 1
                continue

            grouped[cleaned] += 1
            base_counts[row["Base"]] += 1
            raw_points.append(
                {
                    "datetime": row["Date/Time"],
                    "latitude": float(row["Lat"]),
                    "longitude": float(row["Lon"]),
                    "base": row["Base"],
                }
            )

    return grouped, base_counts, pd.DataFrame(raw_points), total_rows, invalid_rows


def grouped_to_frame(grouped):
    columns = [
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
    ]
    rows = [key + (count,) for key, count in grouped.items()]
    df = pd.DataFrame(rows, columns=columns)

    decoded = df["geohash"].apply(geohash.decode)
    df["latitude"] = decoded.apply(lambda pair: pair[0])
    df["longitude"] = decoded.apply(lambda pair: pair[1])
    df = df.sort_values(["year", "month", "day", "time_num", "geohash"])
    return df


def write_base_counts(base_counts, output_dir):
    with (output_dir / "base_counts.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["base", "pickups"])
        for base, count in sorted(base_counts.items()):
            writer.writerow([base, count])


def plot_time_of_day(df, output_dir):
    time_counts = df.groupby("time_cat", as_index=False)["pickups"].sum()
    plt.figure(figsize=(13, 5))
    plt.plot(time_counts["time_cat"], time_counts["pickups"], color="#2563eb", linewidth=2)
    plt.xticks(range(0, len(time_counts), 4), time_counts["time_cat"].iloc[::4], rotation=45)
    plt.xlabel("Time of day")
    plt.ylabel("Pickups")
    plt.title("Uber pickups by 30-minute time bin")
    plt.tight_layout()
    plt.savefig(output_dir / "pickups_by_time.png", dpi=160)
    plt.close()


def plot_day_of_week(df, output_dir):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df.groupby("day_cat")["pickups"].sum().reindex(order).dropna()
    plt.figure(figsize=(9, 5))
    plt.bar(day_counts.index, day_counts.values, color="#0f766e")
    plt.xticks(rotation=25)
    plt.xlabel("Day of week")
    plt.ylabel("Pickups")
    plt.title("Uber pickups by day of week")
    plt.tight_layout()
    plt.savefig(output_dir / "pickups_by_day.png", dpi=160)
    plt.close()


def plot_spatial_heat(df, output_dir):
    spatial = df.groupby(["latitude", "longitude"], as_index=False)["pickups"].sum()
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        spatial["longitude"],
        spatial["latitude"],
        c=spatial["pickups"],
        s=3,
        cmap="inferno",
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Pickups")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Uber pickup density by geohash")
    plt.tight_layout()
    plt.savefig(output_dir / "pickup_density_geohash.png", dpi=180)
    plt.close()


def spatial_bounds(df):
    lon_min = df["longitude"].quantile(0.005)
    lon_max = df["longitude"].quantile(0.995)
    lat_min = df["latitude"].quantile(0.005)
    lat_max = df["latitude"].quantile(0.995)
    lon_pad = (lon_max - lon_min) * 0.04
    lat_pad = (lat_max - lat_min) * 0.04
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def hourly_spatial_frame(df):
    hourly = df.copy()
    hourly["hour"] = hourly["time_cat"].str.slice(0, 2).astype(int)
    return hourly.groupby(["hour", "latitude", "longitude"], as_index=False)["pickups"].sum()


def plot_hourly_pickup_volume(df, output_dir):
    hourly = df.copy()
    hourly["hour"] = hourly["time_cat"].str.slice(0, 2).astype(int)
    counts = hourly.groupby("hour")["pickups"].sum().reindex(range(24), fill_value=0)

    plt.figure(figsize=(13, 5))
    plt.bar(counts.index, counts.values, color="#fb6a4a", width=0.72)
    plt.plot(counts.index, counts.values, color="#a50f15", linewidth=2.2, marker="o", markersize=4)
    plt.xticks(range(24), [f"{hour:02d}:00" for hour in range(24)], rotation=45)
    plt.xlabel("Hour of day")
    plt.ylabel("Pickups")
    plt.title("Uber pickup volume by hour")
    plt.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "pickups_by_hour.png", dpi=170)
    plt.close()


def plot_hourly_density_grid(df, output_dir):
    hourly = hourly_spatial_frame(df)
    lon_min, lon_max, lat_min, lat_max = spatial_bounds(df)
    vmax = max(2, hourly["pickups"].quantile(0.997))
    norm = LogNorm(vmin=1, vmax=vmax)

    fig, axes = plt.subplots(4, 6, figsize=(16, 11), sharex=True, sharey=True)
    for hour, ax in enumerate(axes.flat):
        subset = hourly[hourly["hour"] == hour]
        ax.set_facecolor("#f4f6f8")
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            c=subset["pickups"].clip(lower=1),
            s=4,
            cmap=HEAT_CMAP,
            norm=norm,
            alpha=0.78,
            linewidths=0,
        )
        ax.set_title(f"{hour:02d}:00", fontsize=11)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.tick_params(labelsize=8)
        ax.grid(color="white", linewidth=0.45)

    fig.suptitle("Uber pickup density by hour", fontsize=18, y=0.995)
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")
    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.78, label="Pickups")
    plt.savefig(output_dir / "hourly_density_grid.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_hourly_density_gif(df, output_dir):
    frames_dir = output_dir / "hourly_density_frames"
    frames_dir.mkdir(exist_ok=True)

    hourly = hourly_spatial_frame(df)
    lon_min, lon_max, lat_min, lat_max = spatial_bounds(df)
    vmax = max(2, hourly["pickups"].quantile(0.997))
    norm = LogNorm(vmin=1, vmax=vmax)
    frame_paths = []

    for hour in range(24):
        subset = hourly[hourly["hour"] == hour]
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f2f4f6")
        scatter = ax.scatter(
            subset["longitude"],
            subset["latitude"],
            c=subset["pickups"].clip(lower=1),
            s=8,
            cmap=HEAT_CMAP,
            norm=norm,
            alpha=0.82,
            linewidths=0,
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(color="white", linewidth=0.7)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Pickup density - {hour:02d}:00", loc="left", fontsize=16)
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Pickups")
        plt.tight_layout()

        frame_path = frames_dir / f"hour_{hour:02d}.png"
        plt.savefig(frame_path, dpi=140)
        plt.close()
        frame_paths.append(frame_path)

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    images[0].save(
        output_dir / "hourly_pickup_density.gif",
        save_all=True,
        append_images=images[1:],
        duration=450,
        loop=0,
    )
    for image in images:
        image.close()


def write_summary(df, raw_df, total_rows, invalid_rows, output_dir, dataset_name):
    parsed_datetimes = pd.to_datetime(raw_df["datetime"], format="%m/%d/%Y %H:%M:%S")
    summary = {
        "dataset": dataset_name,
        "source_rows": int(total_rows),
        "valid_rows": int(total_rows - invalid_rows),
        "invalid_rows": int(invalid_rows),
        "aggregated_rows": int(df.shape[0]),
        "total_pickups": int(df["pickups"].sum()),
        "start_datetime": parsed_datetimes.min().strftime("%Y-%m-%d %H:%M:%S"),
        "end_datetime": parsed_datetimes.max().strftime("%Y-%m-%d %H:%M:%S"),
        "unique_geohashes": int(df["geohash"].nunique()),
        "geohash_precision": GEOHASH_PRECISION,
        "time_bins_per_day": TIME_BINS_PER_DAY,
        "minutes_per_bin": MINUTES_PER_BIN,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{dataset_name} Uber Pickup Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #4b5563; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .wide {{ margin-bottom: 18px; }}
    img {{ width: 100%; border: 1px solid #d7dde8; border-radius: 6px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 14px 8px 0; }}
  </style>
</head>
<body>
  <h1>{dataset_name} Uber Pickup Report</h1>
  <div class="meta">Processed with TaxiPrediction-style geohash and 30-minute time-bin aggregation.</div>
  <table>
    <tr><td>Source rows</td><td>{summary["source_rows"]}</td></tr>
    <tr><td>Valid rows</td><td>{summary["valid_rows"]}</td></tr>
    <tr><td>Aggregated rows</td><td>{summary["aggregated_rows"]}</td></tr>
    <tr><td>Total pickups</td><td>{summary["total_pickups"]}</td></tr>
    <tr><td>Time range</td><td>{summary["start_datetime"]} - {summary["end_datetime"]}</td></tr>
    <tr><td>Unique geohashes</td><td>{summary["unique_geohashes"]}</td></tr>
  </table>
  <div class="wide">
    <img src="hourly_pickup_density.gif" alt="Hourly pickup density animation">
  </div>
  <div class="wide">
    <img src="hourly_density_grid.png" alt="Hourly pickup density grid">
  </div>
  <div class="grid">
    <img src="pickups_by_hour.png" alt="Pickups by hour">
    <img src="pickups_by_time.png" alt="Pickups by time">
    <img src="pickups_by_day.png" alt="Pickups by day">
    <img src="pickup_density_geohash.png" alt="Pickup density by geohash">
  </div>
</body>
</html>
"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")


def process_one(dataset_name, input_path):
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped, base_counts, raw_df, total_rows, invalid_rows = aggregate_dataset(input_path)
    df = grouped_to_frame(grouped)

    df.to_csv(output_dir / "taxi_prediction_style_aggregated.csv", index=False)
    raw_df.to_csv(output_dir / "cleaned_pickups.csv", index=False)
    write_base_counts(base_counts, output_dir)
    write_summary(df, raw_df, total_rows, invalid_rows, output_dir, dataset_name)
    plot_time_of_day(df, output_dir)
    plot_day_of_week(df, output_dir)
    plot_spatial_heat(df, output_dir)
    plot_hourly_pickup_volume(df, output_dir)
    plot_hourly_density_grid(df, output_dir)
    plot_hourly_density_gif(df, output_dir)

    return output_dir


def main():
    completed = []
    for dataset_name, input_path in DATASETS.items():
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        completed.append(process_one(dataset_name, input_path))

    print("Created outputs:")
    for path in completed:
        print(path)


if __name__ == "__main__":
    main()
