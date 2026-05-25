#!/usr/bin/env python3
"""TaxiPredict — NYC Uber Taxi Demand Pipeline.

一键运行入口。用法：

    # 全流程
    python pipeline.py

    # 只跑指定步骤
    python pipeline.py --step etl
    python pipeline.py --step features
    python pipeline.py --step train --model xgboost

    # 指定配置
    python pipeline.py --config config/my_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from taxipredict import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TaxiPredict — NYC Uber Taxi Demand Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python pipeline.py                   # 全流程 (--step all --model all)\n"
            "  python pipeline.py --step etl        # 只跑数据层\n"
            "  python pipeline.py --model xgboost   # 只跑 XGBoost 模型\n"
            "  python pipeline.py --step all --model catboost,lightgbm\n"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"TaxiPredict v{__version__}",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径，默认 config/default.yaml",
    )

    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["etl", "features", "analyzer", "train", "all"],
        help="执行哪个阶段（默认 all）",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "模型名或逗号分隔列表 / all。"
            " 可选: xgboost, catboost, lightgbm, gru, lightgcn, stid"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新执行（跳过缓存检查）",
    )

    return parser


# ── 步骤执行函数 ──────────────────────────────────────────────────────────


def _input_csv_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_etl(cfg: dict, force: bool) -> int:
    """ETL 步骤：确保输入数据就绪。

    策略：优先使用已有的 taxi_prediction_hourly_with_weather.csv（旧管线产出）。
    若无，则从原始数据重跑。
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    may_out = processed_dir / "cleaned_may14.csv"
    jun_out = processed_dir / "cleaned_jun14.csv"

    if not force and _input_csv_exists(may_out) and _input_csv_exists(jun_out):
        print(f"跳过 ETL（已存在: {may_out.name}, {jun_out.name}），--force 重新执行")
        return 0

    from taxipredict.etl.loader import safe_read_csv

    # ── 策略 1：直接复制已有的旧管线产出（legacy/ 或原位置） ─────────
    old_may = Path("legacy/new/may14/taxi_prediction_hourly_with_weather.csv")
    if not _input_csv_exists(old_may):
        old_may = Path("new/may14/taxi_prediction_hourly_with_weather.csv")
    old_jun = Path("legacy/new/jun14/taxi_prediction_hourly_with_weather.csv")
    if not _input_csv_exists(old_jun):
        old_jun = Path("new/jun14/taxi_prediction_hourly_with_weather.csv")

    if _input_csv_exists(old_may) and _input_csv_exists(old_jun):
        import shutil
        may_df = safe_read_csv(old_may)
        jun_df = safe_read_csv(old_jun)
        # 统一 datetime 格式
        for df in (may_df, jun_df):
            if "datetime" not in df.columns:
                from taxipredict.etl.loader import build_datetime
                df["datetime"] = build_datetime(df)["datetime"]
        may_df.to_csv(may_out, index=False, encoding="utf-8-sig")
        jun_df.to_csv(jun_out, index=False, encoding="utf-8-sig")
        print(f"ETL: 复用旧管线产出 → {may_out} ({len(may_df)} 行), {jun_out} ({len(jun_df)} 行)")
        return 0

    # ── 策略 2：回退到 style_aggregated 重做聚合 + 天气融合 ──────────
    fallback = Path("legacy/new/may14/taxi_prediction_style_aggregated.csv")
    if not _input_csv_exists(fallback):
        fallback = Path("new/may14/taxi_prediction_style_aggregated.csv")
    if _input_csv_exists(fallback):
        from taxipredict.etl import (
            add_time_features, aggregate_to_hourly, build_datetime,
            join_weather, load_weather,
        )
        print("ETL: 从 style_aggregated 重做聚合 + 天气融合 ...")
        w_csv = "legacy/new/LCD_USW00094728_2014.csv"
        if not Path(w_csv).exists():
            w_csv = "new/LCD_USW00094728_2014.csv"
        weather_path = cfg["data"].get("weather_csv", w_csv)
        weather_df = load_weather(weather_path)

        frames = []
        _agg_may = Path("legacy/new/may14/taxi_prediction_style_aggregated.csv")
        if not _agg_may.exists():
            _agg_may = Path("new/may14/taxi_prediction_style_aggregated.csv")
        _agg_jun = Path("legacy/new/jun14/taxi_prediction_style_aggregated.csv")
        if not _agg_jun.exists():
            _agg_jun = Path("new/jun14/taxi_prediction_style_aggregated.csv")
        for tag, p in [("may14", _agg_may), ("jun14", _agg_jun)]:
            pp = Path(p)
            if not pp.exists():
                continue
            df = safe_read_csv(pp)
            df = build_datetime(df)
            df = add_time_features(df)
            if "pickups" not in df.columns and "order_count" in df.columns:
                df = df.rename(columns={"order_count": "pickups"})
            df["source_file"] = tag
            frames.append(df)
        if not frames:
            print("错误：找不到输入数据", file=sys.stderr)
            return 1
        result = pd.concat(frames, ignore_index=True)
        result = aggregate_to_hourly(result)
        result = join_weather(result, weather_df)

        may_sub = result[result["datetime"].dt.month == 5].copy()
        jun_sub = result[result["datetime"].dt.month == 6].copy()
        may_sub.to_csv(may_out, index=False, encoding="utf-8-sig")
        jun_sub.to_csv(jun_out, index=False, encoding="utf-8-sig")
        print(f"ETL: 回退模式 → {may_out} ({len(may_sub)} 行), {jun_out} ({len(jun_sub)} 行)")
        return 0

    print("错误：找不到任何输入数据", file=sys.stderr)
    return 1


def run_features(cfg: dict, force: bool) -> int:
    """特征工程步骤。"""
    features_dir = Path(cfg["data"]["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)

    may_out = features_dir / "may14_dense_features.csv"
    jun_out = features_dir / "jun14_dense_features.csv"

    if not force and _input_csv_exists(may_out) and _input_csv_exists(jun_out):
        print(f"跳过 Features（已存在），--force 重新执行")
        return 0

    from taxipredict.etl.loader import safe_read_csv
    from taxipredict.features.builder import build_dense_features

    processed_dir = Path(cfg["data"]["processed_dir"])
    feature_cfg = cfg.get("features", {})

    for tag, dst in [("may14", may_out), ("jun14", jun_out)]:
        src = processed_dir / f"cleaned_{tag}.csv"
        if not _input_csv_exists(src):
            print(f"  跳过 {tag}：{src} 不存在")
            continue
        df = safe_read_csv(src)
        dense = build_dense_features(df, feature_cfg)
        dense.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  Features: {dst} ({dense.shape[1]} 列, {len(dense)} 行)")

    return 0


def run_analyzer(cfg: dict) -> int:
    """特征分析步骤。"""
    from taxipredict.features.analyzer import generate_report
    features_dir = Path(cfg["data"]["features_dir"])

    for tag in ("may14", "jun14"):
        src = features_dir / f"{tag}_dense_features.csv"
        dst = features_dir / f"{tag}_feature_report.csv"
        if not _input_csv_exists(src):
            print(f"  跳过 {tag}：{src} 不存在")
            continue
        from taxipredict.etl.loader import safe_read_csv
        df = safe_read_csv(src)
        report = generate_report(df)
        report.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  Analyzer: {dst} ({len(report)} 项特征)")

    return 0


def run_train(cfg: dict, model_names: list[str], force: bool) -> int:
    """训练步骤。"""
    import numpy as np
    import pandas as pd
    from taxipredict.models import get_model_class
    from taxipredict.etl.loader import safe_read_csv
    from taxipredict.etl.split import split_june_last_week

    features_dir = Path(cfg["data"]["features_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = model_dir / "metrics.json"

        if not force and metrics_path.exists():
            print(f"跳过 {model_name}（已存在 {metrics_path}），--force 重新执行")
            continue

        # 读取特征数据
        may_feat = features_dir / "may14_dense_features.csv"
        jun_feat = features_dir / "jun14_dense_features.csv"

        if _input_csv_exists(may_feat) and _input_csv_exists(jun_feat):
            may_df = safe_read_csv(may_feat)
            jun_df = safe_read_csv(jun_feat)
            full_df = pd.concat([may_df, jun_df], ignore_index=True)
        else:
            # 回退：直接用 processed 数据
            may_processed = processed_dir / "cleaned_may14.csv"
            jun_processed = processed_dir / "cleaned_jun14.csv"
            if _input_csv_exists(may_processed) and _input_csv_exists(jun_processed):
                may_df = safe_read_csv(may_processed)
                jun_df = safe_read_csv(jun_processed)
                full_df = pd.concat([may_df, jun_df], ignore_index=True)
            else:
                print(f"  错误：找不到输入数据", file=sys.stderr)
                continue

        # 划分训练/测试
        train_df, test_df = split_june_last_week(full_df)

        # 训练模型
        try:
            ModelClass = get_model_class(model_name)
        except ValueError as e:
            print(f"  {e}")
            continue

        model = ModelClass(cfg)
        try:
            metrics = model.train(train_df, test_df)
        except Exception as e:
            print(f"  {model_name} 失败: {e}")
            continue

        model.save(model_dir / "model.pkl")

        import json
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # ── 生成 outputs ──
        out_cfg = cfg.get("output", {})
        pred_df = getattr(model, "_pred_df", None)
        if pred_df is not None and not pred_df.empty:
            # region_metrics
            from taxipredict.outputs.region_metrics import compute_region_metrics
            region_df = compute_region_metrics(pred_df)
            region_df.to_csv(model_dir / "region_metrics.csv", index=False, encoding="utf-8-sig")
            print(f"  region_metrics: {len(region_df)} 个区域")

            # top 区域预测明细
            top_n = out_cfg.get("top_regions_plot", 10)
            region_df_sorted = region_df.sort_values("r2", ascending=False, na_position="last")
            top_ids = region_df_sorted.head(top_n)["geohash"].tolist()
            top_pred = pred_df[pred_df["geohash"].isin(top_ids)].copy()
            top_pred.to_csv(model_dir / "top_regions_actual_vs_pred_simple.csv",
                            index=False, encoding="utf-8-sig")
            print(f"  top_regions: {len(top_ids)} 个区域预测明细已保存")

            # 对比图
            if out_cfg.get("top_regions_plot", 10) > 0:
                from taxipredict.outputs.visualization import plot_pred_vs_actual
                n_plots = plot_pred_vs_actual(pred_df, model_dir, model_name, top_n=top_n)
                print(f"  对比图: {n_plots} 张")

            # SHAP（仅树模型）
            if out_cfg.get("shap", False) and hasattr(model, "_model") and model._model is not None:
                from taxipredict.outputs.shap_analysis import run_shap
                print(f"  开始 SHAP 分析（{model.__class__.__name__}）...")
                proc_test = getattr(model, "_processed_test", test_df)
                X_test_shap = proc_test[model._feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                # 全表强制数值化（处理 numpy array 嵌套等异常）
                X_test_shap = X_test_shap.apply(pd.to_numeric, errors="coerce").fillna(0)
                run_shap(model._model, X_test_shap, model_dir)

            # Markdown 报告
            if out_cfg.get("report", True):
                from taxipredict.outputs.markdown_report import write_report
                run_config = {
                    "model": model_name,
                    "feature_mode": cfg.get("models", {}).get(model_name, {}).get("feature_mode", "raw_all"),
                    "target": "pickups",
                    "feature_count": len(model._feature_cols),
                    "shap_enabled": bool(out_cfg.get("shap", False)),
                    "top_regions": top_n,
                }
                write_report(model_dir, metrics, model._feature_cols, run_config)
                print(f"  报告: report.md")
        else:
            print(f"  (跳过 outputs: 无 pred_df)")

        print(f"  {model_name} 完成: ", {k: round(v, 4) for k, v in metrics.items() if k.startswith("test_")})

    return 0


def parse_model_arg(raw: str) -> list[str]:
    from taxipredict.models import list_models
    registered = set(list_models())
    raw = raw.strip().lower()
    if raw == "all":
        return list(registered)
    models = [m.strip() for m in raw.split(",")]
    return [m for m in models if m in registered]


def run_pipeline(cfg: dict, args: argparse.Namespace) -> int:
    from taxipredict import __version__
    print(f"TaxiPredict v{__version__} | step={args.step} model={args.model} force={args.force}")

    steps = ["etl", "features", "analyzer", "train"] if args.step == "all" else [args.step]
    models = parse_model_arg(args.model)

    for step in steps:
        print(f"\n{'='*60}")
        print(f"  >>> {step}")
        print(f"{'='*60}")
        if step == "etl":
            code = run_etl(cfg, args.force)
        elif step == "features":
            code = run_features(cfg, args.force)
        elif step == "analyzer":
            code = run_analyzer(cfg)
        elif step == "train":
            code = run_train(cfg, models, args.force)
        else:
            code = 1
        if code:
            return code

    if args.step == "all" and args.model == "all":
        import json, csv
        out_dir = Path(cfg["output"]["dir"])
        summary_path = out_dir / "summary.csv"
        rows = []
        all_keys = set()
        for model_dir in out_dir.iterdir():
            if not model_dir.is_dir():
                continue
            mf = model_dir / "metrics.json"
            if mf.exists():
                d = json.load(open(mf))
                d["model"] = model_dir.name
                rows.append(d)
                all_keys.update(d.keys())
        if rows:
            fieldnames = sorted(all_keys)
            with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
            print(f"\n汇总: {summary_path}")
    return 0


def main() -> int:
    from taxipredict import __version__
    from taxipredict.config import load_config
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    return run_pipeline(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
