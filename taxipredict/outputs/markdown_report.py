"""Markdown 格式实验报告生成。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(
    result_dir: str | Path,
    metrics: dict,
    selected_features: list[str],
    config: dict,
) -> str:
    """生成 Markdown 报告，返回报告路径。"""
    report_path = Path(result_dir) / "report.md"

    lines = [
        f"# {config.get('model_name', 'Model')} 预测报告\n",
    ]

    # 实验设置
    lines.append("## 实验设置\n")
    for key, val in config.items():
        if key in ("model_name", "top_region_ids", "save_full_data"):
            continue
        lines.append(f"- **{key}**: `{val}`")
    lines.append("")

    # 整体指标
    lines.append("## 整体指标\n")
    lines.append("| 数据集 | MAE | RMSE | R² |")
    lines.append("|---|---:|---:|---:|")
    for prefix in ("train", "test"):
        mae = metrics.get(f"{prefix}_mae", "")
        rmse = metrics.get(f"{prefix}_rmse", "")
        r2 = metrics.get(f"{prefix}_r2", "")
        label = "Train" if prefix == "train" else "Test"
        lines.append(f"| {label} | {mae:.6f} | {rmse:.6f} | {r2:.6f} |")
    lines.append("")

    # 特征列表
    lines.append("## 使用特征\n")
    for i, feat in enumerate(selected_features, start=1):
        lines.append(f"{i}. `{feat}`")
    lines.append("")

    # 输出文件说明
    lines.append("## 输出文件\n")
    lines.append("- `metrics.json`：评估指标")
    lines.append("- `run_config.json`：运行配置")
    lines.append("- `region_metrics.csv`：各区域指标")
    lines.append("- `top_regions_actual_vs_pred_simple.csv`：Top 区域预测对比")
    lines.append("- `plots_top_regions/`：各区域对比图")
    if config.get("shap_enabled", False):
        lines.append("- `shap/`：SHAP 分析结果")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return str(report_path)
