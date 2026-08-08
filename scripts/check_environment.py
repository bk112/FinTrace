#!/usr/bin/env python3
"""Report the local training environment without modifying it."""

from __future__ import annotations

import importlib.util
import sys


def installed(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "missing"
    module = __import__(name)
    return getattr(module, "__version__", "installed")


def main() -> int:
    print(f"python={sys.version.split()[0]}")
    for package in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm"):
        print(f"{package}={installed(package)}")

    import torch

    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
