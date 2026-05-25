"""真实值 vs 预测值对比图（按区域）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_pred_vs_actual(
    pred_df: pd.DataFrame,
    result_dir: str | Path,
    model_name: str = "model",
    group_col: str = "geohash",
    target_col: str = "pickups",
    top_n: int = 10,
) -> int:
    """为 top_n 个区域生成真实值 vs 预测值对比图。

    返回生成的图片数量。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    out_dir = Path(result_dir) / "plots_top_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_col not in pred_df.columns:
        return 0

    # 按区域聚合平均绝对误差排序，取前 top_n
    if top_n > 0:
        region_err = pred_df.groupby(group_col)["abs_error"].mean().sort_values()
        top_regions = region_err.head(top_n).index.tolist()
    else:
        top_regions = pred_df[group_col].unique()[:10]

    if "datetime" in pred_df.columns and not pd.api.types.is_datetime64_any_dtype(pred_df["datetime"]):
        pred_df = pred_df.copy()
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"], errors="coerce")

    plot_count = 0
    for region in top_regions:
        region_df = pred_df[pred_df[group_col] == region].sort_values("datetime")
        if len(region_df) < 2:
            continue

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(region_df["datetime"], region_df[target_col], label="Actual", linewidth=2, color="#1f77b4")
        ax.plot(region_df["datetime"], region_df["pred_pickups"], label="Predicted", linewidth=2, color="#ff7f0e", alpha=0.9)

        mae = float(mean_absolute_error(region_df[target_col], region_df["pred_pickups"]))
        ax.set_title(f"{model_name} | {region} (MAE={mae:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Pickups")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        if len(region_df["datetime"]) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            fig.autofmt_xdate()

        plt.tight_layout()
        safe_name = str(region).replace("/", "_").replace("\\", "_")
        plt.savefig(out_dir / f"{safe_name}_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
        plt.close()
        plot_count += 1

    return plot_count


# 需要在这个模块级别导入 sklearn 指标
from sklearn.metrics import mean_absolute_error  # noqa: E402
