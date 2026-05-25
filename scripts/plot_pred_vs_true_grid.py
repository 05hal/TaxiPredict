"""读取精简对比 CSV，输出一张多子图拼接大图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")

COL_ALIASES = {
    "location": ("地点", "location", "geohash", "region"),
    "time": ("时间", "datetime", "time", "timestamp"),
    "pred": ("预测值", "pred", "predicted", "pred_pickups"),
    "true": ("真实值", "true", "actual", "y", "pickups"),
}


def _find_col(columns, aliases):
    for alias in aliases:
        for c in columns:
            if c.lower() == alias.lower():
                return c
    return aliases[0] if aliases else None


def main():
    parser = argparse.ArgumentParser(description="读取精简对比 CSV，输出多子图拼接大图")
    parser.add_argument("--input", type=Path, required=True, help="top_regions_actual_vs_pred_simple.csv 路径")
    parser.add_argument("--output", type=Path, default=None, help="输出 PNG 路径")
    parser.add_argument("--top-n", type=int, default=6, help="展示前 N 个区域（默认 6）")
    parser.add_argument("--figsize", type=int, nargs=2, default=[16, 10], help="画布尺寸")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    loc_col = _find_col(df.columns, COL_ALIASES["location"])
    time_col = _find_col(df.columns, COL_ALIASES["time"])
    pred_col = _find_col(df.columns, COL_ALIASES["pred"])
    true_col = _find_col(df.columns, COL_ALIASES["true"])

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    regions = df[loc_col].unique()[: args.top_n]
    n_cols = min(2, len(regions))
    n_rows = (len(regions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=args.figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, region in enumerate(regions):
        if i >= len(axes):
            break
        rd = df[df[loc_col] == region].sort_values(time_col)
        ax = axes[i]
        ax.plot(rd[time_col], rd[true_col], label="Actual", linewidth=1.5, color="#1f77b4")
        ax.plot(rd[time_col], rd[pred_col], label="Pred", linewidth=1.5, color="#ff7f0e", alpha=0.8)
        mae = mean_absolute_error(rd[true_col], rd[pred_col])
        ax.set_title(f"{region} (MAE={mae:.2f})", fontsize=10)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.autofmt_xdate()

    for i in range(len(regions), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    output_path = args.output or (Path(args.input).parent.parent / "grid_comparison.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
