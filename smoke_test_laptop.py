from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


REQUIRED_IMPORTS = [
    "torch",
    "torchvision",
    "timm",
    "cv2",
    "pytorch_lightning",
    "tensorboardX",
    "skimage",
    "kornia",
    "PIL",
]


def check_imports() -> int:
    failures = 0
    print("== Import check ==")
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[OK]   import {module_name}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] import {module_name}: {type(exc).__name__}: {exc}")
    return failures


def build_model(weights_path: Path, run_forward: bool, input_size: int) -> int:
    print("\n== Model check ==")
    if not weights_path.exists():
        print(
            "[SKIP] model instantiation: pretrained ConvNeXt weights not found at "
            f"{weights_path}"
        )
        return 0

    try:
        import torch
        from model_convnext import fusion_net
    except Exception as exc:
        print(f"[FAIL] model imports: {type(exc).__name__}: {exc}")
        return 1

    try:
        model = fusion_net().cpu().eval()
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(f"[OK]   fusion_net() instantiated on CPU ({total_params:,} parameters)")
    except Exception as exc:
        print(f"[FAIL] fusion_net() instantiation: {type(exc).__name__}: {exc}")
        return 1

    if not run_forward:
        print("[SKIP] CPU forward pass not requested")
        return 0

    try:
        with torch.no_grad():
            sample = torch.randn(1, 3, input_size, input_size)
            output = model(sample)
        print(f"[OK]   CPU forward pass -> output shape {tuple(output.shape)}")
        return 0
    except Exception as exc:
        print(f"[FAIL] CPU forward pass: {type(exc).__name__}: {exc}")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laptop smoke test for DWT-FFC imports and CPU model setup."
    )
    parser.add_argument(
        "--weights-path",
        default="weights/convnext_xlarge_22k_1k_384_ema.pth",
        help="Path to the pretrained ConvNeXt checkpoint used during model instantiation.",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Run a CPU forward pass after model instantiation.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=384,
        help="Square input size for the optional CPU forward pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    weights_path = (root / args.weights_path).resolve()

    failures = 0
    failures += check_imports()
    failures += build_model(
        weights_path=weights_path,
        run_forward=args.forward,
        input_size=args.input_size,
    )

    if failures:
        print(f"\nSmoke test finished with {failures} failure(s).")
        return 1

    print("\nSmoke test finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())