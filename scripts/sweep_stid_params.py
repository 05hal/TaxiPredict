"""STID 参数搜索包装器——循环调用 stid_model.py 的参数组合。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STID_SCRIPT = PROJECT_ROOT / "legacy" / "new_model" / "stid_taxi_demand.py"
if not STID_SCRIPT.exists():
    STID_SCRIPT = PROJECT_ROOT / "new_model" / "stid_taxi_demand.py"


def _parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="STID 参数搜索")
    parser.add_argument("--grid-sizes", default="0.005,0.01,0.02")
    parser.add_argument("--top-regions", default="30,60,100")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--result-root", default=None)
    args = parser.parse_args()

    result_root = (
        Path(args.result_root)
        if args.result_root
        else PROJECT_ROOT / "output" / f"stid_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    result_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for grid_size in _parse_list(args.grid_sizes, float):
        for top_regions in _parse_list(args.top_regions, int):
            name = f"grid{grid_size:g}_top{top_regions}_in{args.input_len}_h{args.horizon}"
            result_dir = result_root / name
            cmd = [
                sys.executable,
                str(STID_SCRIPT),
                "--grid-size", str(grid_size),
                "--top-regions", str(top_regions),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--hidden-dim", str(args.hidden_dim),
                "--embed-dim", str(args.embed_dim),
                "--input-len", str(args.input_len),
                "--horizon", str(args.horizon),
                "--device", args.device,
                "--result-dir", str(result_dir),
            ]
            print(f"\n=== Running {name} ===")
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
            if completed.returncode != 0:
                rows.append({"name": name, "status": "failed", "grid_size": grid_size, "top_regions": top_regions})
            else:
                metrics_path = result_dir / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
                rows.append({"name": name, "status": "ok", "grid_size": grid_size, "top_regions": top_regions, **metrics})

    summary_path = result_root / "sweep_summary.csv"
    if rows:
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n汇总: {summary_path}")


if __name__ == "__main__":
    main()
