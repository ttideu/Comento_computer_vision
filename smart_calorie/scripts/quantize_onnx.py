from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from src.config import DATASET_DIR, MODEL_DIR
from src.data import eval_transform


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, val_dir: Path, batch_size: int, max_batches: int):
        dataset = datasets.ImageFolder(val_dir, transform=eval_transform())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.input_name = input_name
        self.iterator = iter(loader)
        self.max_batches = max_batches
        self.used = 0

    def get_next(self):
        if self.used >= self.max_batches:
            return None
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None
        self.used += 1
        return {self.input_name: images.numpy().astype(np.float32)}


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX FP32 -> INT8 static quantization")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "food_classifier.onnx")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "food_classifier_int8.onnx")
    parser.add_argument("--data", type=Path, default=DATASET_DIR)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    reader = ImageCalibrationReader(
        input_name=input_name,
        val_dir=args.data / "val",
        batch_size=args.batch,
        max_batches=args.calibration_batches,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    fp32_mb = args.model.stat().st_size / (1024 * 1024)
    int8_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"FP32 ONNX : {fp32_mb:.3f} MB")
    print(f"INT8 ONNX : {int8_mb:.3f} MB")
    print(f"size ratio: {int8_mb / fp32_mb:.3f}")
    print(f"saved     : {args.output.resolve()}")


if __name__ == "__main__":
    main()
