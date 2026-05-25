"""共享数据管道：分 geohash 地区、逐小时预测流程。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

LEAKAGE_FEATURE_COLS = {"y_log1p"}
REGION_GROUP_COL = "geohash"
DEFAULT_TOP_REGIONS = 10
DEFAULT_TOP_FEATURES = 20


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk", low_memory=False)


def make_result_dir(base_name: str = "result") -> Path:
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"{base_name}_{time_str}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    required = {"year", "month", "day"}
    if not required.issubset(df.columns):
        raise ValueError("CSV 中必须包含 datetime，或者包含 year/month/day 字段。")

    if "time_cat" in df.columns:
        time_col = "time_cat"
    elif "time" in df.columns:
        time_col = "time"
    else:
        time_col = None

    if time_col is not None:
        df["datetime"] = pd.to_datetime(
            df["year"].astype(str)
            + "-"
            + df["month"].astype(str)
            + "-"
            + df["day"].astype(str)
            + " "
            + df[time_col].astype(str),
            errors="coerce",
        )
    else:
        df["datetime"] = pd.to_datetime(
            df["year"].astype(str)
            + "-"
            + df["month"].astype(str)
            + "-"
            + df["day"].astype(str),
            errors="coerce",
        )

    return df


def recompute_hourly_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute
    time_num = (hour * 60 + minute + 30.0) / (24.0 * 60.0)
    day_of_week = df["datetime"].dt.dayofweek

    df["time_num"] = time_num
    df["time_cos"] = np.cos(time_num * 2 * np.pi)
    df["time_sin"] = np.sin(time_num * 2 * np.pi)
    df["day_cat"] = df["datetime"].dt.day_name()
    df["day_num"] = (day_of_week + time_num) / 7.0
    df["day_cos"] = np.cos(df["day_num"] * 2 * np.pi)
    df["day_sin"] = np.sin(df["day_num"] * 2 * np.pi)
    df["weekend"] = (day_of_week >= 5).astype(int)
    df["hour"] = hour
    df["time_cat"] = hour.map(lambda h: f"{int(h):02d}:00")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    return df


def aggregate_to_hourly_by_geohash(
    df: pd.DataFrame,
    target_col: str = "pickups",
) -> pd.DataFrame:
    """按 geohash + 日期 + 整点小时聚合 pickups（区域×小时需求量）。"""
    if REGION_GROUP_COL not in df.columns:
        raise ValueError("数据必须包含 geohash 列，才能按地区+小时预测。")

    df = df.copy()
    before_rows = len(df)

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour

    group_cols = [REGION_GROUP_COL, "year", "month", "day", "hour"]
    skip_agg = {
        "datetime",
        target_col,
        REGION_GROUP_COL,
        "year",
        "month",
        "day",
        "hour",
    }

    agg_spec: Dict[str, str] = {target_col: "sum"}

    for col in df.columns:
        if col in skip_agg or col in agg_spec:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_spec[col] = "mean"
        else:
            agg_spec[col] = "first"

    out = df.groupby(group_cols, as_index=False).agg(agg_spec)
    out["datetime"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=out["day"], hour=out["hour"])
    )
    out = recompute_hourly_time_features(out)

    region_count = out[REGION_GROUP_COL].nunique()
    slot_count = len(out)
    dup_check = out.duplicated(subset=group_cols).sum()
    if dup_check > 0:
        raise RuntimeError(f"聚合后仍存在重复的地区×小时键：{dup_check} 条")

    print(
        f"\n区域×小时聚合：{before_rows:,} 行 -> {slot_count:,} 行"
        f"（{region_count} 个 geohash，每地区每整点小时一条记录）"
    )

    return out.sort_values([REGION_GROUP_COL, "datetime"]).reset_index(drop=True)


# 兼容旧名称
aggregate_to_hourly_pickups = aggregate_to_hourly_by_geohash


def infer_june_last_week_start(df: pd.DataFrame) -> pd.Timestamp:
    june_df = df[df["datetime"].dt.month == 6].copy()

    if june_df.empty:
        raise ValueError("输入数据中没有 6 月数据，无法划分六月最后一周。")

    max_day = june_df["datetime"].dt.normalize().max()
    return max_day - pd.Timedelta(days=6)


def mark_train_test(df: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["is_test"] = (
        (df["datetime"] >= test_start)
        & (df["datetime"].dt.month == 6)
    ).astype(int)
    df["split"] = np.where(df["is_test"] == 1, "test", "train")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in df.columns:
        if col in ["datetime", "split"]:
            continue

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            encoded_col = f"{col}_encoded"
            le = LabelEncoder()
            df[encoded_col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def drop_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = [c for c in LEAKAGE_FEATURE_COLS if c in df.columns]
    if leakage_cols:
        print(f"\n检测到泄漏特征，已删除：{leakage_cols}")
        df = df.drop(columns=leakage_cols)
    return df


def get_candidate_feature_cols(df: pd.DataFrame, target_col: str = "pickups") -> List[str]:
    exclude_cols = {
        target_col,
        "datetime",
        "split",
        "is_test",
        "geohash",
        "latitude",
        "longitude",
        "time_cat",
        "time",
        "day_cat",
        "source_file",
        *LEAKAGE_FEATURE_COLS,
    }

    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    temp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) < 2:
        return 0.0

    if temp["x"].nunique() <= 1 or temp["y"].nunique() <= 1:
        return 0.0

    value = temp["x"].corr(temp["y"], method=method)
    if pd.isna(value):
        return 0.0

    return float(value)


def rank_features_by_train_correlation(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    rows = []
    y = train_df[target_col]

    for col in feature_cols:
        pearson = safe_corr(train_df[col], y, method="pearson")
        spearman = safe_corr(train_df[col], y, method="spearman")
        selection_score = float(np.mean([abs(pearson), abs(spearman)]))

        rows.append({
            "feature": col,
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "selection_score": selection_score,
        })

    rank_df = pd.DataFrame(rows)
    return rank_df.sort_values("selection_score", ascending=False).reset_index(drop=True)


def evaluate_predictions(y_true, y_pred, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": calc_rmse(y_true, y_pred),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def enrich_prediction_dataframe(pred_df: pd.DataFrame) -> pd.DataFrame:
    """为预测结果补充区域×小时维度字段。"""
    out = pred_df.copy()
    if "datetime" in out.columns:
        out["year"] = out["datetime"].dt.year
        out["month"] = out["datetime"].dt.month
        out["day"] = out["datetime"].dt.day
        out["hour"] = out["datetime"].dt.hour
        out["time_cat"] = out["hour"].map(lambda h: f"{int(h):02d}:00")
    return out


def build_prediction_dataframe(
    test_df: pd.DataFrame,
    y_pred,
    target_col: str,
) -> pd.DataFrame:
    cols = ["datetime", target_col]
    if REGION_GROUP_COL in test_df.columns:
        cols = [REGION_GROUP_COL, "datetime", target_col]
    for extra_col in ("year", "month", "day", "hour", "time_cat", "latitude", "longitude"):
        if extra_col in test_df.columns:
            cols.append(extra_col)

    pred_df = test_df[cols].copy()
    pred_df["pred_pickups"] = y_pred
    pred_df["error"] = pred_df["pred_pickups"] - pred_df[target_col]
    pred_df["abs_error"] = pred_df["error"].abs()
    return enrich_prediction_dataframe(pred_df)


def filter_pred_df_by_regions(
    pred_df: pd.DataFrame,
    region_ids: List[str],
) -> pd.DataFrame:
    if not region_ids:
        return pred_df.iloc[0:0].copy()
    return pred_df[pred_df[REGION_GROUP_COL].isin(region_ids)].copy()


def select_top_regions(
    region_metrics_df: pd.DataFrame,
    n: int = DEFAULT_TOP_REGIONS,
    min_test_rows: int = 48,
) -> List[str]:
    """按测试集 R2 从高到低选取表现最好的 n 个地区。"""
    eligible = region_metrics_df[region_metrics_df["test_rows"] >= min_test_rows].copy()
    if eligible.empty:
        eligible = region_metrics_df.copy()

    ranked = eligible.sort_values(
        ["r2", "mae"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top = ranked.head(n)
    print(f"\n选取测试集表现最好的 {len(top)} 个地区（按 R2 排序）：")
    print(top[["geohash", "test_rows", "mae", "rmse", "r2"]].to_string(index=False))

    return top["geohash"].tolist()


def build_simple_top_regions_export(
    pred_df: pd.DataFrame,
    top_region_ids: List[str],
    target_col: str,
) -> pd.DataFrame:
    """前 N 个地区的精简对比表：地点、时间、预测值、真实值。"""
    top_pred_df = filter_pred_df_by_regions(pred_df, top_region_ids)
    top_pred_df = top_pred_df.sort_values(
        [REGION_GROUP_COL, "datetime"]
    ).reset_index(drop=True)

    return pd.DataFrame({
        "地点": top_pred_df[REGION_GROUP_COL].astype(str),
        "时间": top_pred_df["datetime"],
        "预测值": top_pred_df["pred_pickups"],
        "真实值": top_pred_df[target_col],
    })


def save_top_region_outputs(
    pred_df: pd.DataFrame,
    top_region_ids: List[str],
    target_col: str,
    result_dir: Path,
) -> None:
    """仅保存 top 地区的预测明细 CSV。"""
    top_pred_df = filter_pred_df_by_regions(pred_df, top_region_ids)
    top_pred_df = top_pred_df.sort_values(
        [REGION_GROUP_COL, "datetime"]
    ).reset_index(drop=True)

    top_pred_df.to_csv(
        result_dir / "test_predictions_top_regions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    simple_export = build_simple_top_regions_export(
        pred_df=pred_df,
        top_region_ids=top_region_ids,
        target_col=target_col,
    )
    simple_path = result_dir / "top_regions_actual_vs_pred_simple.csv"
    simple_export.to_csv(simple_path, index=False, encoding="utf-8-sig")
    print(
        f"\n已保存 top {len(top_region_ids)} 地区精简对比表：{simple_path.name}"
        f"（{len(simple_export):,} 行，列：地点/时间/预测值/真实值）"
    )

    pred_by_region_dir = result_dir / "predictions_top_regions"
    pred_by_region_dir.mkdir(parents=True, exist_ok=True)
    for geohash, region_df in top_pred_df.groupby(REGION_GROUP_COL, sort=True):
        region_df.to_csv(
            pred_by_region_dir / f"{safe_geohash_filename(geohash)}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def safe_geohash_filename(geohash: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(geohash))


def compute_region_metrics(
    pred_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    rows = []

    for geohash, region_df in pred_df.groupby(REGION_GROUP_COL, sort=True):
        metrics = evaluate_predictions(
            region_df[target_col],
            region_df["pred_pickups"],
            "region",
        )
        rows.append({
            "geohash": geohash,
            "test_rows": int(len(region_df)),
            "mae": metrics["region_mae"],
            "rmse": metrics["region_rmse"],
            "r2": metrics["region_r2"],
        })

    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def try_run_shap(
    model: Any,
    test_df: pd.DataFrame,
    selected_features: List[str],
    result_dir: Path,
    max_samples: int = 2000,
) -> None:
    """对树模型执行 SHAP 分析，保存重要性表与 summary/bar 图。"""
    shap_dir = result_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n未执行 SHAP：未安装 shap 或 matplotlib。原因：{e}")
        return

    X = test_df[selected_features].copy()

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    print(f"\n开始 SHAP 分析，样本数：{len(X_sample)}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_array = shap_values[0]
    else:
        shap_values_array = shap_values

    mean_abs = np.abs(shap_values_array).mean(axis=0)

    shap_importance = pd.DataFrame({
        "feature": selected_features,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        shap_dir / "shap_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure()
    shap.summary_plot(
        shap_values_array,
        X_sample,
        show=False,
        max_display=min(30, len(selected_features)),
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values_array,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=min(30, len(selected_features)),
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP 结果已保存到：{shap_dir}")


def add_shared_cli_arguments(
    parser: argparse.ArgumentParser,
    result_prefix_default: str = "result",
) -> None:
    parser.add_argument(
        "--inputs",
        nargs=2,
        required=True,
        type=Path,
        help="两个输入 CSV，例如 may14/taxi_prediction_hourly_with_weather.csv jun14/taxi_prediction_hourly_with_weather.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="pickups",
        help="目标列名，默认 pickups",
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        default=result_prefix_default,
        help="输出目录前缀，默认生成 {prefix}_时间 目录",
    )
    parser.add_argument(
        "--top-regions",
        type=int,
        default=DEFAULT_TOP_REGIONS,
        help="仅对测试集 R2 最好的前 N 个 geohash 输出对比图与预测明细，默认 10。",
    )
    parser.add_argument(
        "--save-full-data",
        action="store_true",
        help="若指定，则额外保存 merged_features/train_data/test_data 等大体积 CSV。",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["raw_all", "top_k"],
        default="raw_all",
        help="raw_all=全部特征；top_k=按相关性取前 k 个。默认 raw_all。",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_FEATURES,
        help=f"top_k 模式下使用的特征数，默认 {DEFAULT_TOP_FEATURES}；raw_all 忽略。",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="如果加上该参数，则尝试执行 SHAP 分析。",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=2000,
        help="SHAP 最大采样数，默认 2000。",
    )


def load_input_frames(input_paths: List[Path]) -> pd.DataFrame:
    print("\n读取输入文件：")
    dfs = []

    for path in input_paths:
        print(f" - {path}")
        temp_df = safe_read_csv(path)
        temp_df["source_file"] = str(path)
        dfs.append(temp_df)

    return pd.concat(dfs, axis=0, ignore_index=True)


def prepare_hourly_dataset(
    input_paths: List[Path],
    target_col: str = "pickups",
    feature_mode: str = "raw_all",
    top_k: int = DEFAULT_TOP_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, List[str], pd.DataFrame]:
    df = load_input_frames(input_paths)
    df = drop_leakage_features(df)

    if target_col not in df.columns:
        raise ValueError(f"数据中找不到目标列：{target_col}")

    print(f"\n合并后数据规模：{df.shape}")

    df = build_datetime_column(df)
    df = df.dropna(subset=["datetime"]).copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()

    df = aggregate_to_hourly_by_geohash(df=df, target_col=target_col)

    test_start = infer_june_last_week_start(df)
    df = mark_train_test(df, test_start=test_start)

    print(f"\n自动识别六月最后一周开始时间：{test_start}")
    print("训练集：datetime < test_start")
    print("测试集：datetime >= test_start 且 month == 6")

    df, _ = encode_categorical_columns(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df.sort_values([REGION_GROUP_COL, "datetime"]).reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查输入文件和日期字段。")

    if test_df.empty:
        raise ValueError("测试集为空，请检查六月数据是否包含最后一周。")

    candidate_features = get_candidate_feature_cols(df=df, target_col=target_col)
    ranking_df = rank_features_by_train_correlation(
        train_df=train_df,
        feature_cols=candidate_features,
        target_col=target_col,
    )

    if feature_mode == "raw_all":
        selected_features = candidate_features
        print(f"\n候选特征数量：{len(candidate_features)}")
        print(f"最终使用特征数量：{len(selected_features)}（raw_all：全部特征）")
    else:
        selected_features = ranking_df.head(top_k)["feature"].tolist()
        print(f"\n候选特征数量：{len(candidate_features)}")
        print(f"最终使用特征数量：{len(selected_features)}（top_k：前 {top_k} 个）")
        print("\n入选特征：")
        print(
            ranking_df.head(top_k)[
                ["feature", "selection_score", "pearson_corr", "spearman_corr"]
            ].to_string(index=False)
        )

    print(f"地区数量（geohash）：{df[REGION_GROUP_COL].nunique()}")

    return df, train_df, test_df, test_start, selected_features, ranking_df


def plot_actual_vs_predicted_by_region(
    pred_df: pd.DataFrame,
    target_col: str,
    result_dir: Path,
    model_name: str,
    region_ids: List[str] | None = None,
) -> int:
    """为指定 geohash 地区绘制逐小时真实值 vs 预测值对比图。"""
    if REGION_GROUP_COL not in pred_df.columns:
        raise ValueError("预测结果缺少 geohash 列，无法按地区绘图。")

    if region_ids is not None:
        pred_df = filter_pred_df_by_regions(pred_df, region_ids)

    if pred_df.empty:
        print("\n未生成对比图：没有可绘制的地区。")
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n未生成对比图：未安装 matplotlib。原因：{e}")
        return 0

    plot_dir = result_dir / "plots_top_regions"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_count = 0
    for geohash, region_df in pred_df.groupby(REGION_GROUP_COL, sort=True):
        region_df = region_df.sort_values("datetime")
        x = region_df["datetime"]
        y_true = region_df[target_col].values
        y_pred = region_df["pred_pickups"].values
        region_metrics = evaluate_predictions(y_true, y_pred, "region")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(x, y_true, label="Actual", linewidth=2, color="#1f77b4")
        ax.plot(x, y_pred, label="Predicted", linewidth=2, color="#ff7f0e", alpha=0.9)
        ax.set_title(
            f"{model_name} | geohash={geohash} (hourly demand)\n"
            f"MAE={region_metrics['region_mae']:.2f}, "
            f"RMSE={region_metrics['region_rmse']:.2f}, "
            f"R2={region_metrics['region_r2']:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Datetime (region x hour)", fontsize=11)
        ax.set_ylabel("Pickups per hour", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        if len(x) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            fig.autofmt_xdate()

        plt.tight_layout()
        output_path = plot_dir / f"{safe_geohash_filename(geohash)}_actual_vs_predicted.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        plot_count += 1

    print(f"\n已为 {plot_count} 个地区生成对比图，目录：{plot_dir.name}/")
    return plot_count


def plot_actual_vs_predicted(
    pred_df: pd.DataFrame,
    target_col: str,
    result_dir: Path,
    metrics: Dict,
    model_name: str,
    top_region_ids: List[str],
) -> bool:
    plot_count = plot_actual_vs_predicted_by_region(
        pred_df=pred_df,
        target_col=target_col,
        result_dir=result_dir,
        model_name=model_name,
        region_ids=top_region_ids,
    )
    return plot_count > 0


def write_markdown_report(
    result_dir: Path,
    model_name: str,
    config: Dict,
    metrics: Dict,
    selected_features: List[str],
    test_start: pd.Timestamp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_filename: str,
    top_region_ids: List[str],
    importance_filename: str | None = None,
) -> None:
    report_path = result_dir / "report.md"
    lines = [
        f"# {model_name} 两个月数据预测六月最后一周报告\n",
        "## 1. 实验设置\n",
        f"- 目标变量：`{config['target_col']}`",
        f"- 输入文件数量：`{len(config['input_files'])}`",
        f"- 测试集开始时间：`{str(test_start)}`",
        f"- 训练集样本数：`{len(train_df)}`",
        f"- 测试集样本数：`{len(test_df)}`",
        f"- 特征模式：`{config.get('feature_mode', 'raw_all')}`",
        f"- 实际使用特征数量：`{len(selected_features)}`",
        f"- 预测粒度：`{config.get('prediction_granularity', 'hourly_by_geohash')}`",
        f"- 地区数量（geohash）：`{config.get('region_count', 'N/A')}`",
        f"- 输出对比图/明细的地区数：`{config.get('top_regions', DEFAULT_TOP_REGIONS)}`（按测试 R2 选取）",
    ]
    if config.get("use_topk"):
        lines.append(f"- top-k 特征数：`{config.get('k', DEFAULT_TOP_FEATURES)}`")
    lines.extend([
        "",
        "## 2.1 输出对比图与明细的 top 地区\n",
    ])

    for geohash in top_region_ids:
        lines.append(f"- `{geohash}`")

    lines.extend([
        "",
        "## 2.2 整体评价指标\n",
        "| 数据集 | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|",
        f"| Train | {metrics['train_mae']:.6f} | {metrics['train_rmse']:.6f} | {metrics['train_r2']:.6f} |",
        f"| Test 六月最后一周（全部地区） | {metrics['test_mae']:.6f} | {metrics['test_rmse']:.6f} | {metrics['test_r2']:.6f} |",
        "",
        "## 3. 使用的特征\n",
    ])

    for i, feature in enumerate(selected_features, start=1):
        lines.append(f"{i}. `{feature}`")
    lines.extend([
        "",
        "## 4. 输出文件说明\n",
    ])
    if config.get("save_full_data"):
        lines.extend([
            "- `merged_features.csv`：合并并处理后的完整数据。",
            "- `train_data.csv`：训练集。",
            "- `test_data.csv`：六月最后一周测试集。",
        ])
    lines.extend([
        "- `feature_ranking.csv`：训练集特征相关性排序。",
        "- `selected_features.csv`：最终使用的特征。",
        "- `region_metrics.csv`：全部 geohash 地区的测试指标（按 R2 排序）。",
        "- `test_predictions_top_regions.csv`：top 地区六月最后一周「地区×小时」预测明细。",
        "- `top_regions_actual_vs_pred_simple.csv`：top 地区精简对比（地点、时间、预测值、真实值）。",
        "- `predictions_top_regions/`：top 地区各自的预测明细 CSV。",
        "- `plots_top_regions/`：top 地区的真实值 vs 预测值对比图。",
        "- `metrics.json`：评价指标。",
        "- `run_config.json`：运行参数。",
        f"- `{model_filename}`：训练好的模型。",
    ])

    if importance_filename:
        lines.append(f"- `{importance_filename}`：特征重要性。")

    if config.get("shap_enabled"):
        lines.append("- `shap/`：SHAP 图片与重要性表。")

    lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_standard_outputs(
    result_dir: Path,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics: Dict,
    config: Dict,
    selected_features: List[str],
    ranking_df: pd.DataFrame,
    test_start: pd.Timestamp,
    model_name: str,
    model_filename: str,
    top_regions: int = DEFAULT_TOP_REGIONS,
    save_full_data: bool = False,
    importance_df: pd.DataFrame | None = None,
    importance_filename: str | None = None,
) -> List[str]:
    if save_full_data:
        df.to_csv(result_dir / "merged_features.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(result_dir / "train_data.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(result_dir / "test_data.csv", index=False, encoding="utf-8-sig")

    ranking_df.to_csv(result_dir / "feature_ranking.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": selected_features, "used": 1}).to_csv(
        result_dir / "selected_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    region_metrics_df = compute_region_metrics(
        pred_df=pred_df,
        target_col=config["target_col"],
    )
    region_metrics_df.to_csv(result_dir / "region_metrics.csv", index=False, encoding="utf-8-sig")

    top_region_ids = select_top_regions(
        region_metrics_df=region_metrics_df,
        n=top_regions,
    )
    save_top_region_outputs(
        pred_df=pred_df,
        top_region_ids=top_region_ids,
        target_col=config["target_col"],
        result_dir=result_dir,
    )

    config["top_regions"] = int(top_regions)
    config["top_region_ids"] = top_region_ids
    config["save_full_data"] = bool(save_full_data)
    config.setdefault("feature_mode", "raw_all")
    config.setdefault("k", DEFAULT_TOP_FEATURES)
    config["use_topk"] = config.get("feature_mode") == "top_k"

    save_json(metrics, result_dir / "metrics.json")
    save_json(config, result_dir / "run_config.json")

    plot_actual_vs_predicted(
        pred_df=pred_df,
        target_col=config["target_col"],
        result_dir=result_dir,
        metrics=metrics,
        model_name=model_name,
        top_region_ids=top_region_ids,
    )

    if importance_df is not None and importance_filename:
        importance_df.to_csv(
            result_dir / importance_filename,
            index=False,
            encoding="utf-8-sig",
        )

    write_markdown_report(
        result_dir=result_dir,
        model_name=model_name,
        config=config,
        metrics=metrics,
        selected_features=selected_features,
        test_start=test_start,
        train_df=train_df,
        test_df=test_df,
        model_filename=model_filename,
        top_region_ids=top_region_ids,
        importance_filename=importance_filename,
    )

    return top_region_ids
