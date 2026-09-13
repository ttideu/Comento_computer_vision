from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import DATASET_DIR, MODEL_DIR, REPORT_DIR
from src.data import eval_transform


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX/INT8 모델 test split 평가")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "food_classifier.onnx")
    parser.add_argument("--data", type=Path, default=DATASET_DIR)
    parser.add_argument("--classes", type=Path, default=MODEL_DIR / "classes.json")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--name", type=str, default="onnx_fp32")
    return parser.parse_args()


def main():
    args = parse_args()
    import onnxruntime as ort

    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    test_ds = datasets.ImageFolder(args.data / "test", transform=eval_transform())
    if test_ds.classes != class_names:
        raise RuntimeError(f"class mismatch: {test_ds.classes} != {class_names}")

    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    y_true = []
    y_pred = []
    for images, targets in loader:
        logits = session.run(None, {input_name: images.numpy().astype(np.float32)})[0]
        preds = np.argmax(logits, axis=1)
        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    result = {
        "model": str(args.model),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "test_images": len(y_true),
        "num_classes": len(class_names),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{args.name}_metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"saved: {out.resolve()}")


if __name__ == "__main__":
    main()
