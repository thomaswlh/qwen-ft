"""Print key package versions and a one-device CUDA smoke test."""
from __future__ import annotations

import importlib
import os
import sys


PACKAGES = [
    "verl",
    "torch",
    "ray",
    "vllm",
    "megatron.core",
    "megatron.bridge",
    "peft",
    "datasets",
    "transformers",
    "transfer_queue",
    "flash_attn",
    "transformer_engine",
]


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"OK  {name:24} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            failed.append(name)
            print(f"NO  {name:24} {type(exc).__name__}: {exc}")
    try:
        import torch

        print(f"torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
        print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
        if torch.cuda.is_available():
            x = torch.zeros(1, device="cuda")
            print(f"cuda_ok device={x.device} name={torch.cuda.get_device_name(0)}")
            del x
    except Exception as exc:
        print(f"CUDA smoke failed: {exc}")
        return 1
    optional = {"flash_attn", "transformer_engine"}
    hard = [n for n in failed if n not in optional]
    if hard:
        print(f"missing required imports: {hard}")
        return 1
    print("env check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
