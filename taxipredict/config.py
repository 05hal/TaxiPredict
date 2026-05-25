"""配置加载模块。

用法：
    from taxipredict.config import load_config
    cfg = load_config()                # 加载 config/default.yaml
    cfg = load_config("my/path.yaml")  # 指定配置文件
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _find_project_root() -> Path:
    """向上查找 TaxiPredict 项目根目录（含 pipeline.py 的目录）。"""
    here = Path(__file__).resolve().parent  # taxipredict/
    for parent in [here, here.parent, here.parent.parent]:
        if (parent / "pipeline.py").exists() or (parent / "config").is_dir():
            return parent
    return here.parent  # fallback


PROJECT_ROOT: Path = _find_project_root()
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """加载 YAML 配置，返回字典。

    优先级：
    1. 传入的 config_path
    2. 环境变量 CONFIG_PATH
    3. config/default.yaml
    """
    resolved: Path | None = None

    if config_path is not None:
        resolved = Path(config_path)
    elif "CONFIG_PATH" in os.environ:
        resolved = Path(os.environ["CONFIG_PATH"])

    if resolved is None or not resolved.exists():
        if DEFAULT_CONFIG_PATH.exists():
            resolved = DEFAULT_CONFIG_PATH
        else:
            raise FileNotFoundError(
                f"配置文件不存在：{resolved or DEFAULT_CONFIG_PATH}\n"
                f"请确认 config/default.yaml 存在，或通过 CONFIG_PATH 环境变量指定。"
            )

    with open(resolved, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # 解析后的相对路径转为绝对路径
    cfg.setdefault("data", {})
    for key in ("raw_dir", "weather_csv", "processed_dir", "features_dir"):
        val = cfg["data"].get(key)
        if val:
            p = Path(val)
            if not p.is_absolute():
                cfg["data"][key] = str((PROJECT_ROOT / p).resolve())

    return cfg
