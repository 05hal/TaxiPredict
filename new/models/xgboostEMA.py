# xgboost_two_month_last_week.py
# 功能：
# 1. 从命令行读取两个 CSV，一般是 may14 和 jun14 的 taxi_prediction_hourly_with_weather.csv
# 2. 合并两个月数据
# 3. 自动识别六月最后一周作为测试集
# 4. 用五月 + 六月除最后一周外的数据训练
# 5. 用六月最后一周数据测试
# 6. 支持两种特征模式：
#    - raw_all：只用原始特征，不加时序特征，不做 top-k 筛选
#    - ts_topk：加入少量时序特征，并做 top-k 筛选
# 7. 所有结果保存到 result_时间 目录下

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================
# 1. 基础工具
# =========================

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


def save_json(obj: Dict, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =========================
# 2. 时间字段构造
# =========================

def build_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造 datetime 列。

    优先使用已有 datetime。
    如果没有 datetime，则使用 year + month + day + time_cat / time。
    """

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


# =========================
# 3. 训练集 / 测试集划分
# =========================

def infer_june_last_week_start(df: pd.DataFrame) -> pd.Timestamp:
    """
    自动识别六月最后一周开始时间。

    例如六月最大日期是 2014-06-30，则测试集开始时间是 2014-06-24 00:00:00。
    """

    june_df = df[df["datetime"].dt.month == 6].copy()

    if june_df.empty:
        raise ValueError("输入数据中没有 6 月数据，无法划分六月最后一周。")

    max_day = june_df["datetime"].dt.normalize().max()
    test_start = max_day - pd.Timedelta(days=6)

    return test_start


def mark_train_test(df: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """
    train: datetime < test_start
    test:  datetime >= test_start 且 month == 6
    """

    df = df.copy()

    df["is_test"] = (
        (df["datetime"] >= test_start)
        & (df["datetime"].dt.month == 6)
    ).astype(int)

    df["split"] = np.where(df["is_test"] == 1, "test", "train")

    return df


# =========================
# 4. 类别编码
# =========================

def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    对字符串类别列做 LabelEncoder。
    例如 geohash 会生成 geohash_encoded。

    原始字符串列仍然保留，但不会作为 XGBoost 输入特征。
    """

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


# =========================
# 5. 少量时序特征构造
# =========================

def add_lag_features(
    df: pd.DataFrame,
    observed_col: str,
    group_col: str = "geohash",
    lags: List[int] | None = None,
) -> pd.DataFrame:
    """
    少量 lag 特征。
    不再使用太多 lag，避免时序特征过度膨胀。
    """

    if lags is None:
        lags = [1, 3, 6, 24]

    df = df.copy()

    for lag in lags:
        df[f"pickups_lag_{lag}"] = (
            df.groupby(group_col)[observed_col].shift(lag)
        )

    return df


def add_rolling_mean_features(
    df: pd.DataFrame,
    observed_col: str,
    group_col: str = "geohash",
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    只保留 rolling_mean，不再生成 rolling_std / max / min。
    先 shift(1)，再 rolling，避免当前行 pickups 泄漏。
    """

    if windows is None:
        windows = [3, 6, 24]

    df = df.copy()
    grouped = df.groupby(group_col)[observed_col]

    for window in windows:
        shifted = grouped.shift(1)

        df[f"pickups_rolling_mean_{window}"] = (
            shifted.groupby(df[group_col])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def add_ema_features(
    df: pd.DataFrame,
    observed_col: str,
    group_col: str = "geohash",
    spans: List[int] | None = None,
) -> pd.DataFrame:
    """
    少量 EMA 特征。
    先 shift(1)，避免当前行 pickups 泄漏。
    """

    if spans is None:
        spans = [3, 6, 24]

    df = df.copy()

    for span in spans:
        df[f"pickups_ema_{span}"] = (
            df.groupby(group_col)[observed_col]
            .shift(1)
            .groupby(df[group_col])
            .ewm(span=span, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df


def add_diff_features(
    df: pd.DataFrame,
    observed_col: str,
    group_col: str = "geohash",
) -> pd.DataFrame:
    """
    只保留一个趋势变化特征。
    """

    df = df.copy()

    prev_1 = df.groupby(group_col)[observed_col].shift(1)
    prev_2 = df.groupby(group_col)[observed_col].shift(2)

    df["pickups_diff_1"] = prev_1 - prev_2

    return df


def add_same_hour_history_features(
    df: pd.DataFrame,
    observed_col: str,
    group_col: str = "geohash",
) -> pd.DataFrame:
    """
    只保留同小时 lag，不再保留 same_hour_rolling。
    """

    df = df.copy()

    if "time_cat" in df.columns:
        hour = pd.to_numeric(df["time_cat"].astype(str).str.slice(0, 2), errors="coerce")
        df["hour_for_ts"] = hour
    elif "datetime" in df.columns:
        df["hour_for_ts"] = df["datetime"].dt.hour
    else:
        return df

    same_hour_group = [group_col, "hour_for_ts"]

    df["pickups_same_hour_lag_1"] = (
        df.groupby(same_hour_group)[observed_col].shift(1)
    )

    df = df.drop(columns=["hour_for_ts"])

    return df


def build_limited_time_series_features_no_test_leakage(
    df: pd.DataFrame,
    target_col: str = "pickups",
    group_col: str = "geohash",
) -> pd.DataFrame:
    """
    构造少量时序特征。

    关键处理：
    - 训练集行：observed_pickups = pickups
    - 测试集行：observed_pickups = NaN

    这样测试集的时序特征不会用到六月最后一周的真实 pickups。
    """

    df = df.copy()

    if group_col not in df.columns:
        raise ValueError(f"数据中必须包含 {group_col} 列。")

    df = df.sort_values([group_col, "datetime"]).reset_index(drop=True)

    df["observed_pickups_for_ts"] = np.where(
        df["is_test"] == 1,
        np.nan,
        df[target_col],
    )

    df = add_lag_features(
        df,
        observed_col="observed_pickups_for_ts",
        group_col=group_col,
    )

    df = add_rolling_mean_features(
        df,
        observed_col="observed_pickups_for_ts",
        group_col=group_col,
    )

    df = add_ema_features(
        df,
        observed_col="observed_pickups_for_ts",
        group_col=group_col,
    )

    df = add_diff_features(
        df,
        observed_col="observed_pickups_for_ts",
        group_col=group_col,
    )

    df = add_same_hour_history_features(
        df,
        observed_col="observed_pickups_for_ts",
        group_col=group_col,
    )

    df = df.drop(columns=["observed_pickups_for_ts"])

    return df


def get_time_series_feature_names() -> List[str]:
    return [
        "pickups_lag_1",
        "pickups_lag_3",
        "pickups_lag_6",
        "pickups_lag_24",
        "pickups_rolling_mean_3",
        "pickups_rolling_mean_6",
        "pickups_rolling_mean_24",
        "pickups_ema_3",
        "pickups_ema_6",
        "pickups_ema_24",
        "pickups_diff_1",
        "pickups_same_hour_lag_1",
    ]


# =========================
# 6. 特征选择
# =========================

def get_candidate_feature_cols(df: pd.DataFrame, target_col: str = "pickups") -> List[str]:
    exclude_cols = {
        target_col,
        "datetime",
        "split",
        "is_test",
        "geohash",
        "time_cat",
        "time",
        "day_cat",
        "source_file",
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
    rank_df = rank_df.sort_values("selection_score", ascending=False).reset_index(drop=True)

    return rank_df


def select_top_k_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    k: int,
) -> Tuple[List[str], pd.DataFrame]:
    rank_df = rank_features_by_train_correlation(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    selected_features = rank_df.head(k)["feature"].tolist()

    return selected_features, rank_df


# =========================
# 7. 模型训练与评估
# =========================

def build_xgb_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    random_state: int = 42,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )


def evaluate_predictions(y_true, y_pred, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": calc_rmse(y_true, y_pred),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
    target_col: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> Tuple[XGBRegressor, Dict[str, float], pd.DataFrame]:
    X_train = train_df[selected_features]
    y_train = train_df[target_col]

    X_test = test_df[selected_features]
    y_test = test_df[target_col]

    model = build_xgb_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )

    print("\n开始训练 XGBoost...")
    print(f"训练样本数：{len(X_train)}")
    print(f"测试样本数：{len(X_test)}")
    print(f"使用特征数：{len(selected_features)}")

    model.fit(X_train, y_train, verbose=False)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {}
    metrics.update(evaluate_predictions(y_train, train_pred, "train"))
    metrics.update(evaluate_predictions(y_test, test_pred, "test"))

    pred_df = test_df[["datetime", target_col]].copy()

    if "geohash" in test_df.columns:
        pred_df["geohash"] = test_df["geohash"].values

    if "latitude" in test_df.columns:
        pred_df["latitude"] = test_df["latitude"].values

    if "longitude" in test_df.columns:
        pred_df["longitude"] = test_df["longitude"].values

    pred_df["pred_pickups"] = test_pred
    pred_df["error"] = pred_df["pred_pickups"] - pred_df[target_col]
    pred_df["abs_error"] = pred_df["error"].abs()

    return model, metrics, pred_df


# =========================
# 8. 可选 SHAP 分析
# =========================

def try_run_shap(
    model: XGBRegressor,
    test_df: pd.DataFrame,
    selected_features: List[str],
    result_dir: Path,
    max_samples: int = 2000,
):
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


# =========================
# 9. 报告输出
# =========================

def write_markdown_report(
    result_dir: Path,
    config: Dict,
    metrics: Dict,
    selected_features: List[str],
    test_start: pd.Timestamp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    report_path = result_dir / "report.md"

    lines = []

    lines.append("# XGBoost 两个月数据预测六月最后一周报告\n")
    lines.append("## 1. 实验设置\n")
    lines.append(f"- 目标变量：`{config['target_col']}`")
    lines.append(f"- 特征模式：`{config['feature_mode']}`")
    lines.append(f"- 输入文件数量：`{len(config['input_files'])}`")
    lines.append(f"- 测试集开始时间：`{str(test_start)}`")
    lines.append(f"- 训练集样本数：`{len(train_df)}`")
    lines.append(f"- 测试集样本数：`{len(test_df)}`")
    lines.append(f"- top-k 参数：`{config['k']}`")
    lines.append(f"- 是否执行 top-k 筛选：`{config['use_topk']}`")
    lines.append(f"- 实际使用特征数量：`{len(selected_features)}`")
    lines.append("")

    lines.append("## 2. 评价指标\n")
    lines.append("| 数据集 | MAE | RMSE | R2 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Train | {metrics['train_mae']:.6f} | {metrics['train_rmse']:.6f} | {metrics['train_r2']:.6f} |"
    )
    lines.append(
        f"| Test 六月最后一周 | {metrics['test_mae']:.6f} | {metrics['test_rmse']:.6f} | {metrics['test_r2']:.6f} |"
    )
    lines.append("")

    lines.append("## 3. 使用的特征\n")
    for i, feature in enumerate(selected_features, start=1):
        lines.append(f"{i}. `{feature}`")
    lines.append("")

    lines.append("## 4. 输出文件说明\n")
    lines.append("- `merged_features.csv`：合并并处理后的完整数据。")
    lines.append("- `train_data.csv`：训练集。")
    lines.append("- `test_data.csv`：六月最后一周测试集。")
    lines.append("- `feature_ranking.csv`：如果做 top-k，则为训练集上计算的特征相关性排序；raw_all 模式下也会输出，但不用于筛选。")
    lines.append("- `selected_features.csv`：最终使用的特征。")
    lines.append("- `xgb_feature_importance.csv`：XGBoost 内置特征重要性。")
    lines.append("- `test_predictions.csv`：六月最后一周预测结果。")
    lines.append("- `metrics.json`：评价指标。")
    lines.append("- `run_config.json`：运行参数。")
    lines.append("- `xgb_model.pkl`：训练好的模型。")
    lines.append("- `shap/`：如果安装了 SHAP 且加了 `--shap`，会保存 SHAP 图片和重要性表。")
    lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 10. 主流程
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="使用 may14 + jun14 数据训练 XGBoost，并预测六月最后一周 pickups。"
    )

    parser.add_argument(
        "--inputs",
        nargs=2,
        required=True,
        type=Path,
        help="两个输入 CSV，例如 may14/taxi_prediction_hourly_with_weather.csv jun14/taxi_prediction_hourly_with_weather.csv",
    )

    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=["raw_all", "ts_topk"],
        default="raw_all",
        help="特征模式：raw_all=原始特征不筛选；ts_topk=少量时序特征+top-k筛选。默认 raw_all",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="pickups",
        help="目标列名，默认 pickups",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="ts_topk 模式下只使用前 k 个特征，默认 30；raw_all 模式下忽略该参数。",
    )

    parser.add_argument(
        "--result-prefix",
        type=str,
        default="result",
        help="输出目录前缀，默认 result，会生成 result_时间。",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=800,
        help="XGBoost 树数量，默认 800。",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="学习率，默认 0.03。",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="最大树深度，默认 6。",
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

    args = parser.parse_args()

    result_dir = make_result_dir(args.result_prefix)

    print(f"结果目录：{result_dir}")
    print(f"特征模式：{args.feature_mode}")

    input_paths = [Path(p) for p in args.inputs]

    print("\n读取输入文件：")
    dfs = []

    for p in input_paths:
        print(f" - {p}")
        temp_df = safe_read_csv(p)
        temp_df["source_file"] = str(p)
        dfs.append(temp_df)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    if args.target not in df.columns:
        raise ValueError(f"数据中找不到目标列：{args.target}")

    print(f"\n合并后数据规模：{df.shape}")

    df = build_datetime_column(df)
    df = df.dropna(subset=["datetime"]).copy()

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=[args.target]).copy()

    test_start = infer_june_last_week_start(df)
    df = mark_train_test(df, test_start=test_start)

    print(f"\n自动识别六月最后一周开始时间：{test_start}")
    print("训练集：datetime < test_start")
    print("测试集：datetime >= test_start 且 month == 6")

    if args.feature_mode == "ts_topk":
        df = build_limited_time_series_features_no_test_leakage(
            df=df,
            target_col=args.target,
            group_col="geohash",
        )
        use_topk = True
    else:
        print("\nraw_all 模式：不加入时序特征，不做 top-k 筛选。")
        use_topk = False

    df, encoders = encode_categorical_columns(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = df.sort_values("datetime").reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查输入文件和日期字段。")

    if test_df.empty:
        raise ValueError("测试集为空，请检查六月数据是否包含最后一周。")

    candidate_features = get_candidate_feature_cols(
        df=df,
        target_col=args.target,
    )

    ranking_df = rank_features_by_train_correlation(
        train_df=train_df,
        feature_cols=candidate_features,
        target_col=args.target,
    )

    if args.feature_mode == "raw_all":
        selected_features = candidate_features
    else:
        selected_features = ranking_df.head(args.k)["feature"].tolist()

    selected_df = pd.DataFrame({
        "feature": selected_features,
        "used": 1,
    })

    print(f"\n候选特征数量：{len(candidate_features)}")
    print(f"最终使用特征数量：{len(selected_features)}")

    if args.feature_mode == "ts_topk":
        ts_features = [f for f in selected_features if f in get_time_series_feature_names()]
        raw_features = [f for f in selected_features if f not in get_time_series_feature_names()]
        print(f"其中时序特征数量：{len(ts_features)}")
        print(f"其中原始/编码特征数量：{len(raw_features)}")
        print("\nTop-k 特征：")
        print(ranking_df.head(args.k)[["feature", "selection_score", "pearson_corr", "spearman_corr"]].to_string(index=False))
    else:
        print("\nraw_all 使用所有原始数值特征和类别编码特征，不做筛选。")

    model, metrics, pred_df = train_model(
        train_df=train_df,
        test_df=test_df,
        selected_features=selected_features,
        target_col=args.target,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

    print("\n评估结果：")
    print(f"Train MAE  = {metrics['train_mae']:.6f}")
    print(f"Train RMSE = {metrics['train_rmse']:.6f}")
    print(f"Train R2   = {metrics['train_r2']:.6f}")
    print(f"Test MAE   = {metrics['test_mae']:.6f}")
    print(f"Test RMSE  = {metrics['test_rmse']:.6f}")
    print(f"Test R2    = {metrics['test_r2']:.6f}")

    df.to_csv(result_dir / "merged_features.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(result_dir / "train_data.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(result_dir / "test_data.csv", index=False, encoding="utf-8-sig")

    ranking_df.to_csv(result_dir / "feature_ranking.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(result_dir / "selected_features.csv", index=False, encoding="utf-8-sig")

    xgb_importance = pd.DataFrame({
        "feature": selected_features,
        "xgb_importance": model.feature_importances_,
    }).sort_values("xgb_importance", ascending=False)

    xgb_importance.to_csv(result_dir / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")

    pred_df.to_csv(result_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(model, result_dir / "xgb_model.pkl")

    save_json(metrics, result_dir / "metrics.json")

    config = {
        "input_files": [str(p) for p in input_paths],
        "target_col": args.target,
        "feature_mode": args.feature_mode,
        "k": args.k,
        "use_topk": use_topk,
        "result_dir": str(result_dir),
        "test_start": str(test_start),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "candidate_feature_count": int(len(candidate_features)),
        "selected_feature_count": int(len(selected_features)),
        "limited_time_series_features": get_time_series_feature_names() if args.feature_mode == "ts_topk" else [],
        "shap_enabled": bool(args.shap),
    }

    save_json(config, result_dir / "run_config.json")

    write_markdown_report(
        result_dir=result_dir,
        config=config,
        metrics=metrics,
        selected_features=selected_features,
        test_start=test_start,
        train_df=train_df,
        test_df=test_df,
    )

    if args.shap:
        try_run_shap(
            model=model,
            test_df=test_df,
            selected_features=selected_features,
            result_dir=result_dir,
            max_samples=args.shap_max_samples,
        )

    print("\n全部完成。")
    print(f"所有结果已保存到：{result_dir}")


if __name__ == "__main__":
    main()