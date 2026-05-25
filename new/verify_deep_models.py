"""验证 GRU / LightGCN 依赖是否完整可用。"""
from __future__ import annotations

import sys


def check(label: str, fn) -> None:
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        raise


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    check("tensorflow", lambda: __import__("tensorflow"))
    import tensorflow as tf

    check("tensorflow.keras", lambda: __import__("tensorflow.keras"))
    print(f"       tensorflow {tf.__version__}")

    check("torch", lambda: __import__("torch"))
    import torch

    check("torch.nn", lambda: __import__("torch.nn"))
    check("torch.sparse", lambda: __import__("torch.sparse"))
    print(f"       torch {torch.__version__}  cuda={torch.cuda.is_available()}")

    print("\nGRU 与 LightGCN 依赖验证通过。")


if __name__ == "__main__":
    main()
