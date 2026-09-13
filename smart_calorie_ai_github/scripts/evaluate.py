from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import DATASET_DIR, MODEL_DIR, REPORT_DIR
from src.data import build_dataloaders
from src.model import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="독립 test split 평가")
    parser.add_argument("--data", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "best_model.pth")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, dataset_classes = build_dataloaders(
        args.data, batch_size=args.batch, workers=args.workers
    )
    model, class_names, checkpoint = load_checkpoint(str(args.model), device)

    if dataset_classes != class_names:
        raise RuntimeError(
            f"checkpoint classes와 dataset classes 불일치\n{class_names}\n{dataset_classes}"
        )

    y_true = []
    y_pred = []
    with torch.inference_mode():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=True))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "best_validation_accuracy": checkpoint.get("val_acc"),
        "test_images": len(y_true),
        "num_classes": len(class_names),
        "class_names": class_names,
        "classification_report": report,
    }

    with (REPORT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Food Classification Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"accuracy       : {accuracy:.4f}")
    print(f"macro precision: {metrics['macro_precision']:.4f}")
    print(f"macro recall   : {metrics['macro_recall']:.4f}")
    print(f"macro f1       : {metrics['macro_f1']:.4f}")
    print(f"report         : {(REPORT_DIR / 'test_metrics.json').resolve()}")


if __name__ == "__main__":
    main()
