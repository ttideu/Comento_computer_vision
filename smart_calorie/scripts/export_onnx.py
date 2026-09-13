from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import IMAGE_SIZE, MODEL_DIR
from src.model import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch -> ONNX export")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "best_model.pth")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "food_classifier.onnx")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    model, class_names, _ = load_checkpoint(str(args.model), device)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    with (MODEL_DIR / "classes.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"ONNX: {args.output.resolve()}")
    print(f"classes: {(MODEL_DIR / 'classes.json').resolve()}")


if __name__ == "__main__":
    main()
