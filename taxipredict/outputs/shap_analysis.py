"""SHAP 分析：树模型特征重要性解释。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def run_shap(
    model,
    X_test: pd.DataFrame,
    result_dir: str | Path,
    max_samples: int = 2000,
) -> bool:
    """执行 SHAP 分析，保存 summary 图、bar 图和重要性 CSV。

    成功返回 True，失败（无 shap 库或不支持）返回 False。
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  SHAP 跳过: 缺少依赖 ({e})")
        return False

    shap_dir = Path(result_dir) / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    X = X_test.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    _cat_indices = getattr(model, "cat_feature_indices", None)
    if _cat_indices is not None:
        cat_cols = [list(X.columns)[i] for i in _cat_indices if i < len(X.columns)]
        X_clean = X.copy()
        for col in X_clean.columns:
            if col not in cat_cols:
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce").fillna(0)
            else:
                X_clean[col] = X_clean[col].astype(str).fillna("")
    else:
        X_clean = X.astype({c: float for c in X.columns if pd.api.types.is_numeric_dtype(X[c])})
        for c in X.columns:
            if X[c].dtype == object and c not in X_clean.columns:
                print(f"  跳过非数值列: {c}")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean)
    except Exception as e:
        print(f"  SHAP 分析失败: {e}")
        return False

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 重要性表
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(shap_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")

    max_display = min(30, len(X.columns))

    # summary 图
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # bar 图
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  SHAP 已保存到 {shap_dir}")
    return True
