# feature_correlation_entropy_report.py
# 功能：
# 1. 读取原始 CSV 数据
# 2. 自动识别数值特征、类别特征、布尔特征
# 3. 计算每个特征与 pickups 的相关系数
# 4. 计算每个特征的信息熵
# 5. 输出完整特征分析报告
#
# 输出文件：
# 1. feature_analysis_report.csv
# 2. feature_analysis_report.md
#
# 说明：
# - 数值特征：计算 Pearson 相关系数、Spearman 相关系数
# - 类别特征：先进行类别编码，再计算与 pickups 的相关系数
# - 熵值：
#   - 类别特征按类别分布计算熵
#   - 数值特征先分箱，再计算熵
#
# 注意：
# 类别字段如 geohash 的相关系数只表示“编码后的相关性”，不能直接解释为空间因果关系。
# 对 geohash 这类字段，熵值和唯一值数量更有参考意义。

import os
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def safe_read_csv(file_path: str) -> pd.DataFrame:
    """
    安全读取 CSV 文件。
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到输入文件：{file_path}")

    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")

    return df


def infer_feature_type(series: pd.Series) -> str:
    """
    自动判断特征类型。

    返回：
    - numeric
    - boolean
    - categorical
    - datetime
    - unknown
    """

    s = series.dropna()

    if len(s) == 0:
        return "unknown"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        unique_values = s.nunique()

        if unique_values <= 2:
            return "boolean"

        return "numeric"

    # 尝试判断是否是时间字段
    if series.dtype == "object":
        sample = s.astype(str).head(50)

        datetime_success_count = 0
        for value in sample:
            parsed = pd.to_datetime(value, errors="coerce")
            if not pd.isna(parsed):
                datetime_success_count += 1

        if len(sample) > 0 and datetime_success_count / len(sample) > 0.8:
            return "datetime"

    return "categorical"


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """
    根据频数计算 Shannon 熵。
    """

    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    probabilities = counts / counts.sum()

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def calculate_categorical_entropy(series: pd.Series) -> float:
    """
    类别特征熵值。
    """

    s = series.dropna()

    if len(s) == 0:
        return 0.0

    counts = s.astype(str).value_counts().values

    return shannon_entropy_from_counts(counts)


def calculate_numeric_entropy(series: pd.Series, bins: int = 10) -> float:
    """
    数值特征熵值。

    做法：
    1. 对数值特征进行分箱
    2. 统计每个箱子的样本数量
    3. 计算 Shannon 熵

    默认使用 qcut 等频分箱。
    如果 qcut 失败，则退回 cut 等宽分箱。
    """

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return 0.0

    if s.nunique() <= 1:
        return 0.0

    actual_bins = min(bins, s.nunique())

    try:
        binned = pd.qcut(s, q=actual_bins, duplicates="drop")
    except Exception:
        try:
            binned = pd.cut(s, bins=actual_bins, duplicates="drop")
        except Exception:
            return 0.0

    counts = binned.value_counts().values

    return shannon_entropy_from_counts(counts)


def normalize_entropy(entropy: float, unique_count: int) -> float:
    """
    计算归一化熵。

    原始熵会受到类别数量影响。
    例如 geohash 唯一值很多，熵自然可能较高。

    归一化熵 = entropy / log2(unique_count)

    范围大致为：
    - 0：几乎没有不确定性
    - 1：分布非常均匀
    """

    if unique_count <= 1:
        return 0.0

    max_entropy = np.log2(unique_count)

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def encode_categorical_for_correlation(series: pd.Series) -> pd.Series:
    """
    将类别特征编码成数字，用于粗略计算相关系数。

    注意：
    这种相关系数不能直接解释为类别字段与 pickups 的真实线性关系。
    它只用于粗略排序。
    """

    s = series.astype("category")

    encoded = s.cat.codes.replace(-1, np.nan)

    return encoded.astype(float)


def calculate_pearson_corr(feature: pd.Series, target: pd.Series) -> float:
    """
    计算 Pearson 相关系数。
    """

    temp = pd.DataFrame({
        "feature": feature,
        "target": target,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) < 2:
        return np.nan

    if temp["feature"].nunique() <= 1:
        return np.nan

    if temp["target"].nunique() <= 1:
        return np.nan

    return float(temp["feature"].corr(temp["target"], method="pearson"))


def calculate_spearman_corr(feature: pd.Series, target: pd.Series) -> float:
    """
    计算 Spearman 相关系数。
    Spearman 反映单调关系，比 Pearson 更稳健。
    """

    temp = pd.DataFrame({
        "feature": feature,
        "target": target,
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(temp) < 2:
        return np.nan

    if temp["feature"].nunique() <= 1:
        return np.nan

    if temp["target"].nunique() <= 1:
        return np.nan

    return float(temp["feature"].corr(temp["target"], method="spearman"))


def analyze_single_feature(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    numeric_bins: int = 10,
) -> Dict:
    """
    分析单个特征。
    """

    series = df[feature_col]
    target = pd.to_numeric(df[target_col], errors="coerce")

    feature_type = infer_feature_type(series)

    total_count = len(series)
    missing_count = int(series.isna().sum())
    missing_rate = missing_count / total_count if total_count > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    result = {
        "feature": feature_col,
        "feature_type": feature_type,
        "count": total_count,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "unique_count": unique_count,
        "entropy": np.nan,
        "normalized_entropy": np.nan,
        "pearson_corr": np.nan,
        "spearman_corr": np.nan,
        "abs_pearson_corr": np.nan,
        "abs_spearman_corr": np.nan,
        "importance_score": np.nan,
        "note": "",
    }

    if feature_type in ["numeric", "boolean"]:
        numeric_feature = pd.to_numeric(series, errors="coerce")

        entropy = calculate_numeric_entropy(numeric_feature, bins=numeric_bins)
        normalized = normalize_entropy(entropy, min(numeric_bins, unique_count))

        pearson = calculate_pearson_corr(numeric_feature, target)
        spearman = calculate_spearman_corr(numeric_feature, target)

        result["entropy"] = entropy
        result["normalized_entropy"] = normalized
        result["pearson_corr"] = pearson
        result["spearman_corr"] = spearman
        result["abs_pearson_corr"] = abs(pearson) if not pd.isna(pearson) else np.nan
        result["abs_spearman_corr"] = abs(spearman) if not pd.isna(spearman) else np.nan

    elif feature_type == "categorical":
        entropy = calculate_categorical_entropy(series)
        normalized = normalize_entropy(entropy, unique_count)

        encoded_feature = encode_categorical_for_correlation(series)

        pearson = calculate_pearson_corr(encoded_feature, target)
        spearman = calculate_spearman_corr(encoded_feature, target)

        result["entropy"] = entropy
        result["normalized_entropy"] = normalized
        result["pearson_corr"] = pearson
        result["spearman_corr"] = spearman
        result["abs_pearson_corr"] = abs(pearson) if not pd.isna(pearson) else np.nan
        result["abs_spearman_corr"] = abs(spearman) if not pd.isna(spearman) else np.nan
        result["note"] = "categorical feature encoded before correlation; interpret carefully"

    elif feature_type == "datetime":
        datetime_feature = pd.to_datetime(series, errors="coerce")
        timestamp_feature = datetime_feature.astype("int64") // 10**9
        timestamp_feature = timestamp_feature.replace(-9223372037, np.nan)

        entropy = calculate_categorical_entropy(datetime_feature.astype(str))
        normalized = normalize_entropy(entropy, unique_count)

        pearson = calculate_pearson_corr(timestamp_feature, target)
        spearman = calculate_spearman_corr(timestamp_feature, target)

        result["entropy"] = entropy
        result["normalized_entropy"] = normalized
        result["pearson_corr"] = pearson
        result["spearman_corr"] = spearman
        result["abs_pearson_corr"] = abs(pearson) if not pd.isna(pearson) else np.nan
        result["abs_spearman_corr"] = abs(spearman) if not pd.isna(spearman) else np.nan
        result["note"] = "datetime converted to timestamp before correlation"

    else:
        result["note"] = "unknown or empty feature"

    # 综合重要性分数：
    # 这里不是模型重要性，而是“相关性重要性”。
    # 使用 abs_pearson_corr 和 abs_spearman_corr 的均值。
    corr_values = [
        result["abs_pearson_corr"],
        result["abs_spearman_corr"],
    ]

    corr_values = [v for v in corr_values if not pd.isna(v)]

    if len(corr_values) > 0:
        result["importance_score"] = float(np.mean(corr_values))

    return result


def analyze_features(
    df: pd.DataFrame,
    target_col: str = "pickups",
    numeric_bins: int = 10,
) -> pd.DataFrame:
    """
    分析所有特征。
    """

    if target_col not in df.columns:
        raise ValueError(f"数据中找不到目标列：{target_col}")

    target = pd.to_numeric(df[target_col], errors="coerce")

    if target.isna().all():
        raise ValueError(f"目标列 {target_col} 无法转换为数值，请检查数据。")

    feature_cols = [col for col in df.columns if col != target_col]

    results = []

    for col in feature_cols:
        print(f"正在分析特征：{col}")
        result = analyze_single_feature(
            df=df,
            feature_col=col,
            target_col=target_col,
            numeric_bins=numeric_bins,
        )
        results.append(result)

    report_df = pd.DataFrame(results)

    report_df = report_df.sort_values(
        by="importance_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return report_df


def get_top_features_by_importance(
    report_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    按相关性重要性排序。
    """

    return (
        report_df
        .dropna(subset=["importance_score"])
        .sort_values("importance_score", ascending=False)
        .head(top_k)
    )


def get_top_features_by_entropy(
    report_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    按原始熵值排序。
    """

    return (
        report_df
        .dropna(subset=["entropy"])
        .sort_values("entropy", ascending=False)
        .head(top_k)
    )


def get_top_features_by_normalized_entropy(
    report_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    按归一化熵值排序。
    """

    return (
        report_df
        .dropna(subset=["normalized_entropy"])
        .sort_values("normalized_entropy", ascending=False)
        .head(top_k)
    )


def format_float(value, digits: int = 6) -> str:
    """
    格式化浮点数。
    """

    if pd.isna(value):
        return ""

    return f"{value:.{digits}f}"


def dataframe_to_markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    """
    不依赖 tabulate，手动生成 Markdown 表格。
    """

    if df.empty:
        return "无数据\n"

    lines = []

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines.append(header)
    lines.append(separator)

    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]

            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def write_markdown_report(
    report_df: pd.DataFrame,
    output_path: str,
    input_path: str,
    target_col: str,
    top_k: int = 20,
):
    """
    输出 Markdown 报告。
    """

    top_importance = get_top_features_by_importance(report_df, top_k=top_k)
    top_entropy = get_top_features_by_entropy(report_df, top_k=top_k)
    top_normalized_entropy = get_top_features_by_normalized_entropy(report_df, top_k=top_k)

    importance_cols = [
        "feature",
        "feature_type",
        "importance_score",
        "pearson_corr",
        "spearman_corr",
        "entropy",
        "normalized_entropy",
        "unique_count",
        "missing_rate",
    ]

    entropy_cols = [
        "feature",
        "feature_type",
        "entropy",
        "normalized_entropy",
        "unique_count",
        "importance_score",
        "pearson_corr",
        "spearman_corr",
        "missing_rate",
    ]

    lines = []

    lines.append("# Pickups 特征相关性与熵值分析报告\n")
    lines.append(f"- 输入文件：`{input_path}`")
    lines.append(f"- 目标变量：`{target_col}`")
    lines.append(f"- 特征数量：`{len(report_df)}`")
    lines.append("")

    lines.append("## 1. 指标说明\n")
    lines.append("### 1.1 Pearson 相关系数")
    lines.append("Pearson 相关系数用于衡量特征与 `pickups` 之间的线性相关关系。")
    lines.append("")
    lines.append("### 1.2 Spearman 相关系数")
    lines.append("Spearman 相关系数用于衡量特征与 `pickups` 之间的单调关系，对非线性但单调的关系更稳健。")
    lines.append("")
    lines.append("### 1.3 熵值 entropy")
    lines.append("熵值表示一个特征的信息不确定性或离散程度。熵值越高，说明该字段取值越分散、信息量可能越大。")
    lines.append("")
    lines.append("### 1.4 归一化熵 normalized_entropy")
    lines.append("归一化熵用于削弱唯一值数量对熵值的影响。取值越接近 1，说明该特征分布越均匀。")
    lines.append("")
    lines.append("### 1.5 importance_score")
    lines.append("这里的 `importance_score` 不是模型训练得到的重要性，而是相关性重要性：")
    lines.append("")
    lines.append("```text")
    lines.append("importance_score = mean(abs(Pearson), abs(Spearman))")
    lines.append("```")
    lines.append("")

    lines.append(f"## 2. 与 pickups 相关性最高的前 {top_k} 个特征\n")
    lines.append(dataframe_to_markdown_table(top_importance, importance_cols))

    lines.append(f"## 3. 原始熵值最高的前 {top_k} 个特征\n")
    lines.append(dataframe_to_markdown_table(top_entropy, entropy_cols))

    lines.append(f"## 4. 归一化熵值最高的前 {top_k} 个特征\n")
    lines.append(dataframe_to_markdown_table(top_normalized_entropy, entropy_cols))

    lines.append("## 5. 分析建议\n")
    lines.append("1. `importance_score` 高的字段，通常应优先保留，因为它们与 `pickups` 的统计关系更强。")
    lines.append("2. 熵值高但相关性低的字段，不一定没用，可能需要通过模型学习非线性关系。")
    lines.append("3. `geohash` 这类类别空间字段，直接编码后的相关系数解释性较弱，建议结合 XGBoost 特征重要性或 SHAP 再判断。")
    lines.append("4. 如果后续加入 `lag / rolling / EMA` 特征，建议再次运行本脚本，比较时序特征是否进入相关性前列。")
    lines.append("5. 如果某字段缺失率很高，即使相关性较强，也需要谨慎使用。")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_to_console(report_df: pd.DataFrame, top_k: int = 10):
    """
    在控制台输出简要结果。
    """

    print("\n" + "=" * 80)
    print(f"相关性重要性 Top {top_k}")
    print("=" * 80)

    cols = [
        "feature",
        "feature_type",
        "importance_score",
        "pearson_corr",
        "spearman_corr",
        "entropy",
        "normalized_entropy",
        "unique_count",
    ]

    top_importance = get_top_features_by_importance(report_df, top_k=top_k)
    print(top_importance[cols].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"原始熵值 Top {top_k}")
    print("=" * 80)

    top_entropy = get_top_features_by_entropy(report_df, top_k=top_k)
    print(top_entropy[cols].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"归一化熵值 Top {top_k}")
    print("=" * 80)

    top_normalized_entropy = get_top_features_by_normalized_entropy(report_df, top_k=top_k)
    print(top_normalized_entropy[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="计算 CSV 特征与 pickups 的相关系数和熵值，并输出报告。"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV 文件路径，例如 taxi_prediction_style_aggregated.csv",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="pickups",
        help="目标变量列名，默认 pickups",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="feature_analysis_report.csv",
        help="输出 CSV 报告路径",
    )

    parser.add_argument(
        "--output_md",
        type=str,
        default="feature_analysis_report.md",
        help="输出 Markdown 报告路径",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="报告中展示前多少个特征",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="数值特征计算熵时的分箱数量",
    )

    args = parser.parse_args()

    print(f"读取数据：{args.input}")
    df = safe_read_csv(args.input)

    print(f"数据规模：{df.shape}")
    print(f"字段数量：{len(df.columns)}")

    if args.target not in df.columns:
        raise ValueError(
            f"目标列 `{args.target}` 不存在。当前 CSV 字段为：{list(df.columns)}"
        )

    print(f"目标变量：{args.target}")

    report_df = analyze_features(
        df=df,
        target_col=args.target,
        numeric_bins=args.bins,
    )

    report_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\nCSV 报告已保存到：{args.output_csv}")

    write_markdown_report(
        report_df=report_df,
        output_path=args.output_md,
        input_path=args.input,
        target_col=args.target,
        top_k=args.top_k,
    )
    print(f"Markdown 报告已保存到：{args.output_md}")

    write_summary_to_console(
        report_df=report_df,
        top_k=min(args.top_k, 10),
    )

    print("\n分析完成。")


if __name__ == "__main__":
    main()

# python feature_correlation_entropy_report.py --input taxi_prediction_hourly_with_weather_may.csv