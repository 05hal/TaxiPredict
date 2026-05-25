"""读取精简对比 CSV（地点/时间/预测值/真实值），输出一张多子图拼接大图。"""

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


def configure_chinese_font() -> None:
    """尽量启用 Windows 常见中文字体，避免中文标签乱码。"""
    from matplotlib import font_manager

    candidates = ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()

COL_ALIASES = {
    "location": ("地点", "location", "geohash", "region"),
    "time": ("时间", "datetime", "time", "timestamp"),
    "pred": ("预测值", "pred", "predicted", "pred_pickups"),
    "true": ("真实值", "true", "actual", "y", "pickups"),
}


def resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ValueError(f"找不到列（候选：{', '.join(aliases)}），当前列：{list(df.columns)}")


def read_simple_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk", low_memory=False)

    location_col = resolve_column(df, COL_ALIASES["location"])
    time_col = resolve_column(df, COL_ALIASES["time"])
    pred_col = resolve_column(df, COL_ALIASES["pred"])
    true_col = resolve_column(df, COL_ALIASES["true"])

    out = pd.DataFrame(
        {
            "location": df[location_col].astype(str),
            "time": pd.to_datetime(df[time_col], errors="coerce"),
            "pred": pd.to_numeric(df[pred_col], errors="coerce"),
            "true": pd.to_numeric(df[true_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["location", "time", "pred", "true"])
    return out.sort_values(["location", "time"]).reset_index(drop=True)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(r2_score(y_true, y_pred)),
    )


def choose_grid(n_regions: int) -> tuple[int, int]:
    if n_regions <= 0:
        raise ValueError("没有可绘制的地区。")
    ncols = min(5, n_regions)
    nrows = int(np.ceil(n_regions / ncols))
    return nrows, ncols


def plot_pred_vs_true_grid(
    df: pd.DataFrame,
    output_path: Path,
    title: str | None = None,
    dpi: int = 200,
) -> Path:
    locations = df["location"].drop_duplicates().tolist()
    nrows, ncols = choose_grid(len(locations))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        sharex=False,
        squeeze=False,
    )

    for idx, location in enumerate(locations):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        region_df = df[df["location"] == location]
        x = region_df["time"]
        y_true = region_df["true"].values
        y_pred = region_df["pred"].values
        mae, rmse, r2 = calc_metrics(y_true, y_pred)

        ax.plot(x, y_true, label="真实值", linewidth=1.8, color="#1f77b4")
        ax.plot(x, y_pred, label="预测值", linewidth=1.8, color="#ff7f0e", alpha=0.9)
        ax.set_title(
            f"{location}\nMAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("时间", fontsize=9)
        ax.set_ylabel("需求量", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        if len(x) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            for label in ax.get_xticklabels():
                label.set_rotation(25)
                label.set_ha("right")

    for idx in range(len(locations), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 top_regions_actual_vs_pred_simple.csv，输出 pred vs true 拼接大图。"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="输入 CSV 路径（列：地点/时间/预测值/真实值）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出图片路径，默认与输入 CSV 同目录下的 top_regions_pred_vs_true_grid.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Top 地区预测值 vs 真实值",
        help="整张图标题",
    )
    parser.add_argument("--dpi", type=int, default=200, help="输出 DPI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"找不到输入文件：{args.input}")

    df = read_simple_csv(args.input)
    output_path = args.output or (args.input.parent / "top_regions_pred_vs_true_grid.png")
    plot_pred_vs_true_grid(df, output_path=output_path, title=args.title, dpi=args.dpi)

    n_regions = df["location"].nunique()
    print(f"已读取 {len(df):,} 行，共 {n_regions} 个地点。")
    print(f"已保存拼接大图：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
