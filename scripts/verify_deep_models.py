#!/usr/bin/env python3
"""验证深度学习依赖是否可用。"""

from __future__ import annotations

import importlib
import sys

CHECKS = [
    ("pandas", None),
    ("numpy", None),
    ("scikit-learn", "sklearn"),
    ("xgboost", None),
    ("lightgbm", None),
    ("catboost", None),
    ("tensorflow", None),
    ("torch", None),
]


def main():
    all_ok = True
    for name, import_name in CHECKS:
        try:
            mod = importlib.import_module(import_name or name)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅ {name:20s} {ver}")
        except ImportError:
            print(f"  ❌ {name:20s} 未安装")
            all_ok = False
    print()
    if all_ok:
        print("全部依赖可用。")
    else:
        print("部分依赖缺失。运行相应的 requirements/*.txt 安装。")
        sys.exit(1)


if __name__ == "__main__":
    main()
