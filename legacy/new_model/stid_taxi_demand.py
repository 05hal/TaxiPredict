from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    print(
        "Missing dependency. Please install dependencies first:\n"
        "  pip install -r new/models/requirements_stid.txt",
        file=sys.stderr,
    )
    raise

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    class _MissingNN:
        Module = object

    nn = _MissingNN()
    DataLoader = None
    Dataset = object

from pandas.tseries.holiday import USFederalHolidayCalendar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = PROJECT_ROOT / "new_model"
DEFAULT_MAY = PROJECT_ROOT / "data_preprocess" / "data" / "uber-raw-data-may14.csv"
DEFAULT_JUN = PROJECT_ROOT / "data_preprocess" / "data" / "uber-raw-data-jun14.csv"
DEFAULT_WEATHER = PROJECT_ROOT / "new" / "LCD_USW00094728_2014.csv"


@dataclass
class RunConfig:
    order_csvs: list[str]
    weather_csv: str
    result_dir: str
    freq: str
    grid_size: float
    top_regions: int
    input_len: int
    horizon: int
    train_ratio: float
    val_ratio: float
    epochs: int
    batch_size: int
    hidden_dim: int
    embed_dim: int
    num_layers: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Run:\n"
            "  pip install -r new/models/requirements_stid.txt"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("T", "0", regex=False)
        .str.replace("s", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", low_memory=False)


def build_region_id(lat: pd.Series, lon: pd.Series, grid_size: float) -> pd.Series:
    lat_bucket = np.floor(lat.astype(float) / grid_size).astype("Int64")
    lon_bucket = np.floor(lon.astype(float) / grid_size).astype("Int64")
    return lat_bucket.astype(str) + "_" + lon_bucket.astype(str)


def load_orders(paths: Iterable[Path], freq: str, grid_size: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = safe_read_csv(path)
        required = {"Date/Time", "Lat", "Lon", "Base"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        use = df[["Date/Time", "Lat", "Lon", "Base"]].copy()
        use["dt"] = pd.to_datetime(use["Date/Time"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        use["lat"] = pd.to_numeric(use["Lat"], errors="coerce")
        use["lon"] = pd.to_numeric(use["Lon"], errors="coerce")
        use = use.dropna(subset=["dt", "lat", "lon", "Base"])
        use["time_bin"] = use["dt"].dt.floor(freq)
        use["region_id"] = build_region_id(use["lat"], use["lon"], grid_size)
        frames.append(use[["time_bin", "region_id", "Base"]])

    if not frames:
        raise ValueError("No order files were provided")
    return pd.concat(frames, ignore_index=True)


def load_weather(weather_csv: Path, freq: str) -> pd.DataFrame:
    w = safe_read_csv(weather_csv)
    if "DATE" not in w.columns:
        raise ValueError("Weather CSV must contain DATE")

    w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
    w = w[w["DATE"].notna()].copy()
    if "REPORT_TYPE" in w.columns:
        hourly = w[w["REPORT_TYPE"].isin(["FM-15", "FM-16"])].copy()
        if not hourly.empty:
            w = hourly

    w["time_bin"] = w["DATE"].dt.floor(freq)
    w = w.sort_values(["time_bin", "DATE"]).groupby("time_bin", as_index=False).tail(1)

    rename = {
        "HourlyDryBulbTemperature": "temp",
        "HourlyRelativeHumidity": "humidity",
        "HourlyWindSpeed": "wind_speed",
        "HourlyVisibility": "visibility",
        "HourlyPrecipitation": "precip",
        "HourlyStationPressure": "pressure",
    }
    for src in rename:
        if src not in w.columns:
            w[src] = np.nan

    weather = w[["time_bin", *rename.keys()]].rename(columns=rename).copy()
    for col in ["temp", "humidity", "wind_speed", "visibility", "precip", "pressure"]:
        weather[col] = clean_numeric(weather[col])

    weather["is_precip"] = (weather["precip"].fillna(0) > 0).astype(float)
    weather["low_visibility"] = (weather["visibility"] < 5).fillna(False).astype(float)
    weather["high_wind"] = (weather["wind_speed"] > 10).fillna(False).astype(float)
    return weather


def add_time_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame({"time_bin": times})
    out["hour"] = out["time_bin"].dt.hour
    out["weekday"] = out["time_bin"].dt.weekday
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(float)
    out["is_morning_peak"] = out["hour"].between(7, 9).astype(float)
    out["is_evening_peak"] = out["hour"].between(16, 19).astype(float)
    out["is_peak"] = ((out["is_morning_peak"] == 1) | (out["is_evening_peak"] == 1)).astype(float)

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=times.min().normalize(), end=times.max().normalize())
    out["is_holiday"] = out["time_bin"].dt.normalize().isin(holidays).astype(float)

    tod = (out["hour"].astype(float) + 0.5) / 24.0
    dow = (out["weekday"].astype(float) + tod) / 7.0
    out["tod_sin"] = np.sin(2 * math.pi * tod)
    out["tod_cos"] = np.cos(2 * math.pi * tod)
    out["dow_sin"] = np.sin(2 * math.pi * dow)
    out["dow_cos"] = np.cos(2 * math.pi * dow)
    return out


def make_dense_panel(
    orders: pd.DataFrame,
    weather: pd.DataFrame,
    freq: str,
    top_regions: int,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    region_rank = orders.groupby("region_id").size().sort_values(ascending=False)
    regions = region_rank.head(top_regions).index.tolist()
    orders = orders[orders["region_id"].isin(regions)].copy()

    base_counts = (
        orders.groupby(["time_bin", "region_id", "Base"]).size().unstack(fill_value=0).reset_index()
    )
    base_cols = [c for c in base_counts.columns if c not in {"time_bin", "region_id"}]
    base_counts = base_counts.rename(columns={c: f"base_{c}" for c in base_cols})
    base_cols = [f"base_{c}" for c in base_cols]

    demand = orders.groupby(["time_bin", "region_id"]).size().rename("demand").reset_index()

    full_times = pd.date_range(
        orders["time_bin"].min().floor(freq),
        orders["time_bin"].max().floor(freq),
        freq=freq,
    )
    index = pd.MultiIndex.from_product([full_times, regions], names=["time_bin", "region_id"])
    panel = index.to_frame(index=False)
    panel = panel.merge(demand, on=["time_bin", "region_id"], how="left")
    panel = panel.merge(base_counts, on=["time_bin", "region_id"], how="left")
    panel["demand"] = panel["demand"].fillna(0).astype(float)
    for col in base_cols:
        panel[col] = panel[col].fillna(0).astype(float)

    time_features = add_time_features(full_times)
    panel = panel.merge(time_features, on="time_bin", how="left")
    panel = panel.merge(weather, on="time_bin", how="left")

    weather_cols = ["temp", "humidity", "wind_speed", "visibility", "precip", "pressure"]
    flag_cols = ["is_precip", "low_visibility", "high_wind"]
    panel = panel.sort_values(["region_id", "time_bin"]).reset_index(drop=True)
    for col in weather_cols:
        panel[col] = panel[col].ffill().bfill()
    for col in flag_cols:
        panel[col] = panel[col].fillna(0)

    exog_cols = [
        "tod_sin",
        "tod_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_morning_peak",
        "is_evening_peak",
        "is_peak",
        "is_holiday",
        *weather_cols,
        *flag_cols,
    ]
    for col in [*exog_cols, *base_cols]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0).astype(float)

    return panel, regions, exog_cols, base_cols


def panel_to_arrays(
    panel: pd.DataFrame,
    regions: list[str],
    exog_cols: list[str],
    base_cols: list[str],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    demand = panel.pivot(index="time_bin", columns="region_id", values="demand").reindex(columns=regions)
    demand = demand.fillna(0).astype(float)

    exog_by_time = panel.groupby("time_bin", as_index=True)[exog_cols].mean().reindex(demand.index)
    exog_by_time = exog_by_time.ffill().bfill().fillna(0).astype(float)

    base_arrays = []
    for col in base_cols:
        base = panel.pivot(index="time_bin", columns="region_id", values=col).reindex(index=demand.index, columns=regions)
        base_arrays.append(base.fillna(0).values.astype(np.float32))
    if base_arrays:
        base_hist = np.stack(base_arrays, axis=-1)
    else:
        base_hist = np.zeros((len(demand.index), len(regions), 0), dtype=np.float32)

    return demand.index, demand.values.astype(np.float32), exog_by_time.values.astype(np.float32), base_hist


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean_.reshape(-1).tolist(), "std": self.std_.reshape(-1).tolist()}


class DemandWindowDataset(Dataset):
    def __init__(
        self,
        demand: np.ndarray,
        base_hist: np.ndarray,
        exog: np.ndarray,
        input_len: int,
        horizon: int,
        start_index: int,
        end_index: int,
    ) -> None:
        self.demand = demand
        self.base_hist = base_hist
        self.exog = exog
        self.input_len = input_len
        self.horizon = horizon
        self.starts = list(range(start_index, end_index - input_len - horizon + 1))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        hist_end = s + self.input_len
        target_end = hist_end + self.horizon
        demand_hist = self.demand[s:hist_end].T
        base_hist = self.base_hist[s:hist_end].transpose(1, 0, 2).reshape(self.demand.shape[1], -1)
        x_hist = np.concatenate([demand_hist, base_hist], axis=1)
        x_exog = self.exog[hist_end:target_end]
        y = self.demand[hist_end:target_end].T
        return (
            torch.from_numpy(x_hist.astype(np.float32)),
            torch.from_numpy(x_exog.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class STIDTaxiDemand(nn.Module):
    """STID-style MLP with spatial identity, time identity and external features."""

    def __init__(
        self,
        num_nodes: int,
        hist_dim: int,
        horizon: int,
        exog_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.02)
        self.exog_proj = nn.Linear(horizon * exog_dim, embed_dim)

        in_dim = hist_dim + embed_dim + embed_dim
        self.encoder = MLP(in_dim, hidden_dim, num_layers, dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x_hist, x_exog):
        batch_size, num_nodes, _ = x_hist.shape
        exog_flat = x_exog.reshape(batch_size, -1)
        exog_emb = self.exog_proj(exog_flat).unsqueeze(1).expand(-1, num_nodes, -1)
        node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat([x_hist, node_emb, exog_emb], dim=-1)
        encoded = self.encoder(features)
        return self.head(encoded)


def split_indices(total_steps: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    if train_end <= 48 or val_end <= train_end:
        raise ValueError("Not enough time steps for the requested train/val split")
    return train_end, val_end


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": r2}


def inverse_scaled_demand(arr: np.ndarray, scaler: StandardScaler, num_nodes: int) -> np.ndarray:
    by_horizon = arr.transpose(0, 2, 1)
    restored = scaler.inverse_transform(by_horizon.reshape(-1, num_nodes))
    return restored.reshape(by_horizon.shape).transpose(0, 2, 1)


def run_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    losses: list[float] = []
    loss_fn = nn.SmoothL1Loss()
    for x_hist, x_exog, y in loader:
        x_hist = x_hist.to(device)
        x_exog = x_exog.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_hist, x_exog)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def evaluate(model, loader, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for x_hist, x_exog, y in loader:
        x_hist = x_hist.to(device)
        x_exog = x_exog.to(device)
        y = y.to(device)
        pred = model(x_hist, x_exog)
        losses.append(float(loss_fn(pred, y).detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds), np.concatenate(trues)


def save_predictions(
    path: Path,
    times: pd.DatetimeIndex,
    regions: list[str],
    input_len: int,
    horizon: int,
    test_start: int,
    pred: np.ndarray,
    true: np.ndarray,
) -> None:
    rows = []
    sample_starts = list(range(test_start, len(times) - input_len - horizon + 1))
    for sample_idx, s in enumerate(sample_starts):
        first_target = s + input_len
        for node_idx, region in enumerate(regions):
            for h in range(horizon):
                rows.append(
                    {
                        "time_bin": times[first_target + h],
                        "region_id": region,
                        "horizon": h + 1,
                        "y_true": float(true[sample_idx, node_idx, h]),
                        "y_pred": float(pred[sample_idx, node_idx, h]),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an STID model for regional taxi demand.")
    parser.add_argument(
        "--order-csvs",
        nargs="+",
        default=[str(DEFAULT_MAY), str(DEFAULT_JUN)],
        help="Raw Uber order CSV files. Default: May 2014 and June 2014 files in the repo.",
    )
    parser.add_argument("--weather-csv", default=str(DEFAULT_WEATHER), help="NOAA/LCD weather CSV.")
    parser.add_argument("--result-dir", default=None, help="Directory for outputs.")
    parser.add_argument("--freq", default="1h", help="Aggregation frequency, e.g. 1h or 30min.")
    parser.add_argument("--grid-size", type=float, default=0.01, help="Lat/lon grid size for regions.")
    parser.add_argument("--top-regions", type=int, default=80, help="Keep busiest regions to control memory.")
    parser.add_argument("--input-len", type=int, default=24, help="Historical steps used by STID.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon steps.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    set_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    result_dir = (
        Path(args.result_dir)
        if args.result_dir
        else NEW_ROOT / "models" / f"stid_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    order_paths = [Path(p) for p in args.order_csvs]
    weather_path = Path(args.weather_csv)

    print("Loading and aggregating orders...")
    orders = load_orders(order_paths, args.freq, args.grid_size)
    print(f"Order rows after cleaning: {len(orders):,}")

    print("Loading weather...")
    weather = load_weather(weather_path, args.freq)

    print("Building dense region-time panel...")
    panel, regions, exog_cols, base_cols = make_dense_panel(orders, weather, args.freq, args.top_regions)
    times, demand_raw, exog_raw, base_raw = panel_to_arrays(panel, regions, exog_cols, base_cols)
    train_end, val_end = split_indices(len(times), args.train_ratio, args.val_ratio)

    demand_log = np.log1p(demand_raw)
    base_log = np.log1p(base_raw)
    demand_scaler = StandardScaler().fit(demand_log[:train_end])
    exog_scaler = StandardScaler().fit(exog_raw[:train_end])
    base_scaler = StandardScaler().fit(base_log[:train_end].reshape(-1, max(len(base_cols), 1)))
    demand = demand_scaler.transform(demand_log).astype(np.float32)
    exog = exog_scaler.transform(exog_raw).astype(np.float32)
    base_hist = base_scaler.transform(base_log.reshape(-1, max(len(base_cols), 1))).reshape(base_log.shape).astype(np.float32)

    train_ds = DemandWindowDataset(demand, base_hist, exog, args.input_len, args.horizon, 0, train_end)
    val_ds = DemandWindowDataset(demand, base_hist, exog, args.input_len, args.horizon, train_end - args.input_len, val_end)
    test_ds = DemandWindowDataset(demand, base_hist, exog, args.input_len, args.horizon, val_end - args.input_len, len(times))
    if min(len(train_ds), len(val_ds), len(test_ds)) <= 0:
        raise ValueError("Empty train/val/test dataset. Reduce input_len/horizon or adjust split ratios.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = STIDTaxiDemand(
        num_nodes=len(regions),
        hist_dim=args.input_len * (1 + len(base_cols)),
        horizon=args.horizon,
        exog_dim=len(exog_cols),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    stale = 0
    history = []
    print(
        f"Training STID on {device}: regions={len(regions)}, steps={len(times)}, "
        f"exog_dim={len(exog_cols)}, historical_base_dims={len(base_cols)}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_pred_scaled, test_true_scaled = evaluate(model, test_loader, device)
    pred_log = inverse_scaled_demand(test_pred_scaled, demand_scaler, len(regions))
    true_log = inverse_scaled_demand(test_true_scaled, demand_scaler, len(regions))
    pred = np.expm1(pred_log).clip(min=0)
    true = np.expm1(true_log).clip(min=0)
    metrics = masked_metrics(true, pred)

    config = RunConfig(
        order_csvs=[str(p) for p in order_paths],
        weather_csv=str(weather_path),
        result_dir=str(result_dir),
        freq=args.freq,
        grid_size=args.grid_size,
        top_regions=args.top_regions,
        input_len=args.input_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "regions": regions,
            "exog_cols": exog_cols,
            "base_cols": base_cols,
            "demand_scaler": demand_scaler.to_dict(),
            "exog_scaler": exog_scaler.to_dict(),
            "base_scaler": base_scaler.to_dict(),
        },
        result_dir / "stid_model.pt",
    )
    panel.to_csv(result_dir / "training_panel.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    save_predictions(result_dir / "test_predictions.csv", times, regions, args.input_len, args.horizon, val_end - args.input_len, pred, true)

    with (result_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    with (result_dir / "run_config.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, ensure_ascii=False, indent=2)
    with (result_dir / "feature_columns.json").open("w", encoding="utf-8") as fp:
        json.dump({"regions": regions, "exog_cols": exog_cols, "base_cols": base_cols}, fp, ensure_ascii=False, indent=2)

    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nSaved outputs to: {result_dir}")


if __name__ == "__main__":
    main()
