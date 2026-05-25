"""CSV 加载与日期时间解析。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """安全读取 CSV，自动处理 utf-8 / gbk 编码。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到输入文件：{p.resolve()}")
    try:
        return pd.read_csv(p, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk", low_memory=False)


def _resolve_input_paths(
    may_path: str | Path | None,
    jun_path: str | Path | None,
    cfg: dict | None,
) -> tuple[Path, Path]:
    """确定双月 CSV 路径：优先参数，其次配置，最后报错。"""
    from taxipredict.config import load_config

    if cfg is None:
        cfg = load_config()

    def _resolve(p: str | Path | None, cfg_key: str) -> Path:
        if p is not None:
            return Path(p)
        # 从配置的 raw_dir 推断
        raw_dir = Path(cfg.get("data", {}).get("raw_dir", ""))
        if raw_dir.exists():
            candidates = list(raw_dir.glob(f"{cfg_key}*"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"找不到 {cfg_key} 数据。请指定路径或将其放入 {raw_dir}"
        )

    return _resolve(may_path, "uber-raw-data-may14"), _resolve(jun_path, "uber-raw-data-jun14")


def load_month_pair(
    may_path: str | Path | None = None,
    jun_path: str | Path | None = None,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """加载双月数据合并为一张表，标记 source_file 来源。"""
    may_p, jun_p = _resolve_input_paths(may_path, jun_path, cfg)
    print(f"加载数据：{may_p.name} + {jun_p.name}")

    frames = []
    for p, tag in [(may_p, "may14"), (jun_p, "jun14")]:
        df = safe_read_csv(p)
        df["source_file"] = tag
        frames.append(df)
        print(f"  {tag}: {len(df)} 行, {df.shape[1]} 列")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    print(f"合并后: {len(combined)} 行")
    return combined


def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """构造 datetime 列。

    优先级：
    1. 已有 datetime 列
    2. year + month + day + time_cat / time
    3. Date/Time 原始字段（Uber 格式："5/6/2014 0:01:00"）
    """
    out = df.copy()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        return out

    if "Date/Time" in out.columns:
        out["datetime"] = pd.to_datetime(out["Date/Time"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        invalid = out["datetime"].isna().sum()
        if invalid > 0:
            print(f"  警告：{invalid} 行日期解析失败，将被丢弃")
        return out

    # 从 year / month / day / time_cat 合成
    required = {"year", "month", "day"}
    if not required.issubset(out.columns):
        raise ValueError("数据缺少 datetime 或 Date/Time 或 year/month/day 列")

    time_col = "time_cat" if "time_cat" in out.columns else ("time" if "time" in out.columns else None)
    if time_col:
        out["datetime"] = pd.to_datetime(
            out["year"].astype(str) + "-"
            + out["month"].astype(str) + "-"
            + out["day"].astype(str) + " "
            + out[time_col].astype(str),
            errors="coerce",
        )
    else:
        out["datetime"] = pd.to_datetime(
            out["year"].astype(str) + "-"
            + out["month"].astype(str) + "-"
            + out["day"].astype(str),
            errors="coerce",
        )

    invalid = out["datetime"].isna().sum()
    if invalid > 0:
        print(f"  警告：{invalid} 行日期解析失败，将被丢弃")
    return out
