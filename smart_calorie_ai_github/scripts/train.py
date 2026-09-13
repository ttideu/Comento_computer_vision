from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import DATASET_DIR, MODEL_DIR, REPORT_DIR
from src.data import build_dataloaders
from src.model import build_model, freeze_feature_extractor, unfreeze_all


def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV3-Small Food-101 fine-tuning")
    parser.add_argument("--data", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, _, class_names = build_dataloaders(
        args.data, batch_size=args.batch, workers=args.workers
    )
    print("classes:", class_names)

    model = build_model(len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best_val_acc = -1.0
    checkpoint_path = MODEL_DIR / "best_model.pth"

    # Stage 1: classifier head만 먼저 적응
    freeze_feature_extractor(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.warmup_epochs, 1))

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            # Stage 2: 전체 네트워크를 낮은 학습률로 fine-tuning
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.25, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
            )
            print("[Stage 2] feature extractor unfreeze")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "architecture": "mobilenet_v3_small",
                    "image_size": 224,
                },
                checkpoint_path,
            )
            print(f"  -> best 저장: {checkpoint_path} ({best_val_acc:.4f})")

    with (REPORT_DIR / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n학습 완료")
    print(f"best val accuracy: {best_val_acc:.4f}")
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
