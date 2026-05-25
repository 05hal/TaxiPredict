# GRU 分 geohash 地区、区域×小时 pickups 预测
# 数据管道与 xgboostEMA 一致：去泄漏、区域×小时聚合、top-N 地区输出

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from taxi_pipeline import (
    REGION_GROUP_COL,
    add_shared_cli_arguments,
    build_prediction_dataframe,
    evaluate_predictions,
    make_result_dir,
    prepare_hourly_dataset,
    save_standard_outputs,
)

warnings.filterwarnings("ignore")

MODEL_NAME = "GRU"


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, dt = [], [], []

    for i in range(window_size, len(features)):
        X.append(features[i - window_size : i])
        y.append(target[i])
        dt.append(datetimes[i])

    return np.array(X), np.array(y), np.array(dt)


def train_gru(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list[str],
    target_col: str,
    window_size: int,
    epochs: int,
    batch_size: int,
) -> tuple[Sequential, StandardScaler, dict, pd.DataFrame]:
    timeline = pd.concat([train_df, test_df], ignore_index=True).sort_values(
        [REGION_GROUP_COL, "datetime"]
    )
    test_start_dt = test_df["datetime"].min()

    scaler = StandardScaler()
    scaler.fit(timeline[selected_features])

    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    test_meta: list[dict] = []

    for geohash, region_df in timeline.groupby(REGION_GROUP_COL, sort=False):
        region_df = region_df.sort_values("datetime")
        features = scaler.transform(region_df[selected_features]).astype(np.float32)
        target_values = region_df[target_col].values.astype(np.float32)
        datetimes = region_df["datetime"].values

        if len(region_df) <= window_size:
            continue

        X_all, y_all, dt_all = create_sequences(
            features,
            target_values,
            datetimes,
            window_size,
        )
        is_test = dt_all >= np.datetime64(test_start_dt)

        if (~is_test).any():
            X_train_parts.append(X_all[~is_test])
            y_train_parts.append(y_all[~is_test])

        if is_test.any():
            test_indices = np.where(is_test)[0]
            X_test_parts.append(X_all[is_test])
            y_test_parts.append(y_all[is_test])
            for idx in test_indices:
                test_meta.append({
                    REGION_GROUP_COL: geohash,
                    "datetime": pd.to_datetime(dt_all[idx]),
                    target_col: float(y_all[idx]),
                })

    if not X_train_parts or not X_test_parts:
        raise ValueError("GRU 序列样本不足，请检查各地区数据量或减小 --window-size。")

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print(f"\n开始训练 {MODEL_NAME}...")
    print(f"训练序列数：{len(X_train)}")
    print(f"测试序列数：{len(X_test)}")
    print(f"窗口大小：{window_size}")
    print(f"使用特征数：{len(selected_features)}")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    train_pred = model.predict(X_train, verbose=0).flatten()
    test_pred = np.maximum(model.predict(X_test, verbose=0).flatten(), 0)

    metrics = {}
    metrics.update(evaluate_predictions(y_train, train_pred, "train"))
    metrics.update(evaluate_predictions(y_test, test_pred, "test"))

    test_eval_df = pd.DataFrame(test_meta)
    pred_df = build_prediction_dataframe(test_eval_df, test_pred, target_col)

    return model, scaler, metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 may14 + jun14 数据训练 GRU，并预测六月最后一周 pickups。"
    )
    add_shared_cli_arguments(parser, result_prefix_default="result_gru")
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="GRU 时间窗口大小（小时），默认 24。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="训练轮数，默认 30。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批大小，默认 32。",
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

    model, scaler, metrics, pred_df = train_gru(
        train_df=train_df,
        test_df=test_df,
        selected_features=selected_features,
        target_col=args.target,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n评估结果：")
    print(f"Train MAE  = {metrics['train_mae']:.6f}")
    print(f"Train RMSE = {metrics['train_rmse']:.6f}")
    print(f"Train R2   = {metrics['train_r2']:.6f}")
    print(f"Test MAE   = {metrics['test_mae']:.6f}")
    print(f"Test RMSE  = {metrics['test_rmse']:.6f}")
    print(f"Test R2    = {metrics['test_r2']:.6f}")

    joblib.dump(
        {"model": model, "scaler": scaler, "features": selected_features},
        result_dir / "gru_model.pkl",
    )

    config = {
        "model": MODEL_NAME,
        "input_files": [str(p) for p in input_paths],
        "target_col": args.target,
        "result_dir": str(result_dir),
        "test_start": str(test_start),
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_feature_count": int(len(selected_features)),
        "prediction_granularity": "hourly_by_geohash",
        "region_count": int(df[REGION_GROUP_COL].nunique()),
        "feature_mode": args.feature_mode,
        "k": args.k,
    }

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
        model_filename="gru_model.pkl",
        top_regions=args.top_regions,
        save_full_data=args.save_full_data,
    )

    print("\n全部完成。")
    print(f"所有结果已保存到：{result_dir}")


if __name__ == "__main__":
    main()
