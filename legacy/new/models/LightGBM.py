# LightGBM 分 geohash 地区、区域×小时 pickups 预测
# 数据管道与 xgboostEMA 一致：去泄漏、区域×小时聚合、top-N 地区输出

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi_pipeline import (
    REGION_GROUP_COL,
    add_shared_cli_arguments,
    build_prediction_dataframe,
    evaluate_predictions,
    make_result_dir,
    prepare_hourly_dataset,
    save_standard_outputs,
    try_run_shap,
)

warnings.filterwarnings("ignore")

MODEL_NAME = "LightGBM"


def train_lightgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    target_col: str,
    num_boost_round: int,
    learning_rate: float,
) -> tuple[lgb.Booster, dict, pd.DataFrame]:
    X_train = train_df[selected_features]
    y_train = train_df[target_col]
    X_test = test_df[selected_features]
    y_test = test_df[target_col]

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    print(f"\n开始训练 {MODEL_NAME}...")
    print(f"训练样本数：{len(X_train)}")
    print(f"测试样本数：{len(X_test)}")
    print(f"使用特征数：{len(selected_features)}")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_pred = np.maximum(test_pred, 0)

    metrics = {}
    metrics.update(evaluate_predictions(y_train, train_pred, "train"))
    metrics.update(evaluate_predictions(y_test, test_pred, "test"))

    pred_df = build_prediction_dataframe(test_df, test_pred, target_col)
    return model, metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 may14 + jun14 数据训练 LightGBM，并预测六月最后一周 pickups。"
    )
    add_shared_cli_arguments(parser, result_prefix_default="result_lgb")
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=2000,
        help="LightGBM 迭代轮数，默认 2000。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="学习率，默认 0.03。",
    )
    args = parser.parse_args()

    result_dir = make_result_dir(args.result_prefix)
    input_paths = [Path(p) for p in args.inputs]

    print(f"结果目录：{result_dir}")

    print(f"特征模式：{args.feature_mode}")

    df, train_df, test_df, test_start, selected_features, ranking_df = prepare_hourly_dataset(
        input_paths=input_paths,
        target_col=args.target,
        feature_mode=args.feature_mode,
        top_k=args.k,
    )

    model, metrics, pred_df = train_lightgbm(
        train_df=train_df,
        test_df=test_df,
        selected_features=selected_features,
        target_col=args.target,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
    )

    print("\n评估结果：")
    print(f"Train MAE  = {metrics['train_mae']:.6f}")
    print(f"Train RMSE = {metrics['train_rmse']:.6f}")
    print(f"Train R2   = {metrics['train_r2']:.6f}")
    print(f"Test MAE   = {metrics['test_mae']:.6f}")
    print(f"Test RMSE  = {metrics['test_rmse']:.6f}")
    print(f"Test R2    = {metrics['test_r2']:.6f}")

    importance_df = pd.DataFrame({
        "feature": selected_features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    config = {
        "model": MODEL_NAME,
        "input_files": [str(p) for p in input_paths],
        "target_col": args.target,
        "result_dir": str(result_dir),
        "test_start": str(test_start),
        "num_boost_round": args.num_boost_round,
        "learning_rate": args.learning_rate,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_features)),
        "prediction_granularity": "hourly_by_geohash",
        "region_count": int(df[REGION_GROUP_COL].nunique()),
        "feature_mode": args.feature_mode,
        "k": args.k,
        "shap_enabled": bool(args.shap),
    }

    joblib.dump(model, result_dir / "lgb_model.pkl")

    save_standard_outputs(
        result_dir=result_dir,
        df=df,
        train_df=train_df,
        test_df=test_df,
        pred_df=pred_df,
        metrics=metrics,
        config=config,
        selected_features=selected_features,
        ranking_df=ranking_df,
        test_start=test_start,
        model_name=MODEL_NAME,
        model_filename="lgb_model.pkl",
        top_regions=args.top_regions,
        save_full_data=args.save_full_data,
        importance_df=importance_df,
        importance_filename="lgb_feature_importance.csv",
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
