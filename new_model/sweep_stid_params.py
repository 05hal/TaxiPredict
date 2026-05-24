from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "new_model" / "stid_taxi_demand.py"


def parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def run_one(
    python_exe: str,
    result_root: Path,
    grid_size: float,
    top_regions: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    embed_dim: int,
    input_len: int,
    horizon: int,
    device: str,
) -> dict:
    name = f"grid{grid_size:g}_top{top_regions}_in{input_len}_h{horizon}"
    result_dir = result_root / name
    cmd = [
        python_exe,
        str(SCRIPT),
        "--grid-size",
        str(grid_size),
        "--top-regions",
        str(top_regions),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--hidden-dim",
        str(hidden_dim),
        "--embed-dim",
        str(embed_dim),
        "--input-len",
        str(input_len),
        "--horizon",
        str(horizon),
        "--device",
        device,
        "--result-dir",
        str(result_dir),
    ]

    print(f"\n=== Running {name} ===")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        return {
            "name": name,
            "grid_size": grid_size,
            "top_regions": top_regions,
            "input_len": input_len,
            "horizon": horizon,
            "status": "failed",
            "result_dir": str(result_dir),
        }

    metrics_path = result_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "grid_size": grid_size,
        "top_regions": top_regions,
        "input_len": input_len,
        "horizon": horizon,
        "status": "ok",
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape_percent": metrics.get("mape_percent"),
        "r2": metrics.get("r2"),
        "result_dir": str(result_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep STID grid_size and top_regions.")
    parser.add_argument("--grid-sizes", default="0.005,0.01,0.02")
    parser.add_argument("--top-regions", default="30,60,100")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--result-root", default=None)
    args = parser.parse_args()

    result_root = (
        Path(args.result_root)
        if args.result_root
        else PROJECT_ROOT / "new_model" / f"stid_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for grid_size in parse_list(args.grid_sizes, float):
        for top_regions in parse_list(args.top_regions, int):
            row = run_one(
                python_exe=sys.executable,
                result_root=result_root,
                grid_size=grid_size,
                top_regions=top_regions,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                input_len=args.input_len,
                horizon=args.horizon,
                device=args.device,
            )
            rows.append(row)
            write_summary(result_root / "sweep_summary.csv", rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        best = min(ok_rows, key=lambda r: (r["rmse"], r["mae"]))
        print("\nBest by RMSE:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\nSweep summary saved to: {result_root / 'sweep_summary.csv'}")


def write_summary(path: Path, rows: list[dict]) -> None:
    columns = [
        "name",
        "status",
        "grid_size",
        "top_regions",
        "input_len",
        "horizon",
        "mae",
        "rmse",
        "mape_percent",
        "r2",
        "result_dir",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


if __name__ == "__main__":
    main()
