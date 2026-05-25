"""所有模型注册表。"""

from __future__ import annotations

from typing import Dict, Type

from taxipredict.models.base import BaseModel
from taxipredict.models.xgboost_model import XGBoostModel
from taxipredict.models.catboost_model import CatBoostModel
from taxipredict.models.lightgbm_model import LightGBMModel

_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
}

# GRU（可选，需要 TensorFlow）
try:
    from taxipredict.models.gru_model import GRUModel  # noqa: E402
    _MODEL_REGISTRY["gru"] = GRUModel
except Exception:
    pass

# LightGCN（可选，需要 PyTorch）
try:
    from taxipredict.models.lightgcn_model import LightGCNModel  # noqa: E402
    _MODEL_REGISTRY["lightgcn"] = LightGCNModel
except Exception:
    pass

# STID（可选，需要 PyTorch + 原始数据）
try:
    import torch  # noqa: F401
    from taxipredict.models.stid_model import STIDModel  # noqa: E402
    _MODEL_REGISTRY["stid"] = STIDModel
except Exception:
    pass


def register_model(name: str, cls: Type[BaseModel]) -> None:
    _MODEL_REGISTRY[name] = cls


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可用: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY.keys())
