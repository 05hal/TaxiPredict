"""STID 模型适配器——包装 new_model/stid_taxi_demand.py。

数据流：
1. 优先使用 pipeline ETL 产出的 processed/features 数据
2. 回退：加载原始 Uber CSV（保持 old STID 行为）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from taxipredict.models.base import BaseModel

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class STIDModel(BaseModel):
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self._model = None
        self._config = {}
        self._result_dir = None

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，STID 需要 torch>=2.1")

        from taxipredict.config import PROJECT_ROOT

        stid_cfg = self.cfg.get("models", {}).get("stid", {})
        # 优先从 legacy/ 找，若不存在则从原位置找（兼容旧 repo 结构）
        stid_script = PROJECT_ROOT / "legacy" / "new_model" / "stid_taxi_demand.py"
        if not stid_script.exists():
            stid_script = PROJECT_ROOT / "new_model" / "stid_taxi_demand.py"
        if not stid_script.exists():
            raise FileNotFoundError(f"STID 脚本不存在: {stid_script}")

        # 构建输出目录
        output_dir = Path(self.cfg["output"]["dir"]) / "stid"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检查能否使用 pipeline 已处理的数据直接构建 panel
        # 如果 train/test 已经包含所有需要的信息，可以直接调用 STID 内部逻辑
        # 但 STID 需要原始坐标做网格聚合，而 train/test 可能只有 geohash
        # → 回退到旧 STID 脚本：通过子进程调用

        import subprocess

        order_csvs = [
            str(PROJECT_ROOT / "data_preprocess" / "data" / "uber-raw-data-may14.csv"),
            str(PROJECT_ROOT / "data_preprocess" / "data" / "uber-raw-data-jun14.csv"),
        ]
        weather_csv = self.cfg["data"].get("weather_csv", str(PROJECT_ROOT / "new" / "LCD_USW00094728_2014.csv"))

        cmd = [
            sys.executable,
            str(stid_script),
            "--order-csvs", *order_csvs,
            "--weather-csv", str(weather_csv),
            "--result-dir", str(output_dir / "result"),
            "--grid-size", str(stid_cfg.get("grid_size", 0.02)),
            "--top-regions", str(stid_cfg.get("top_regions", 30)),
            "--input-len", str(stid_cfg.get("input_len", 24)),
            "--horizon", str(stid_cfg.get("horizon", 1)),
            "--epochs", str(stid_cfg.get("epochs", 40)),
            "--batch-size", str(stid_cfg.get("batch_size", 32)),
            "--hidden-dim", str(stid_cfg.get("hidden_dim", 128)),
            "--embed-dim", str(stid_cfg.get("embed_dim", 32)),
            "--num-layers", str(stid_cfg.get("num_layers", 2)),
            "--dropout", str(stid_cfg.get("dropout", 0.1)),
            "--learning-rate", str(stid_cfg.get("learning_rate", 0.001)),
            "--weight-decay", str(stid_cfg.get("weight_decay", 0.0001)),
            "--patience", str(stid_cfg.get("patience", 12)),
            "--device", str(stid_cfg.get("device", "auto")),
        ]

        print(f"  STID: 调用旧脚本 {stid_script.name} ...")
        print(f"  STID: 原始数据路径需存在: {order_csvs[0]}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.returncode != 0:
            print(f"  STID 错误: {result.stderr[-500:]}", file=sys.stderr)
            return {"status": "failed", "returncode": result.returncode}

        # 解析指标
        metrics = {}
        metrics_candidates = list(output_dir.rglob("metrics.json"))
        if metrics_candidates:
            with open(metrics_candidates[0]) as f:
                metrics = json.load(f)
        else:
            metrics = {"status": "completed", "note": "metrics.json not found in expected path"}

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("STID predict 需要单独调用旧脚本")

    def save(self, path: str | Path) -> None:
        pass  # STID 自己保存

    @classmethod
    def load(cls, path: str | Path) -> "STIDModel":
        model = cls.__new__(cls)
        return model
