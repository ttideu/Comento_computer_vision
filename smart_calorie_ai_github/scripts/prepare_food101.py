from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import DATASET_DIR, TARGET_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Food-101 10-class subset builder")
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def ensure_dirs(root: Path):
    for split in ("train", "val", "test"):
        for cls in TARGET_CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def save_train(root: Path, count_per_class: int, quality: int):
    ds = load_dataset("ethz/food101", split="train", streaming=True)
    counts = defaultdict(int)
    target = set(TARGET_CLASSES)
    total_needed = count_per_class * len(target)
    saved = 0

    pbar = tqdm(total=total_needed, desc="train 저장")
    for sample in ds:
        label_name = ds.features["label"].int2str(sample["label"])
        if label_name not in target or counts[label_name] >= count_per_class:
            continue

        idx = counts[label_name]
        out = root / "train" / label_name / f"{label_name}_{idx:04d}.jpg"
        sample["image"].convert("RGB").save(out, quality=quality)
        counts[label_name] += 1
        saved += 1
        pbar.update(1)

        if saved >= total_needed:
            break
    pbar.close()

    missing = {c: count_per_class - counts[c] for c in target if counts[c] < count_per_class}
    if missing:
        raise RuntimeError(f"train 데이터 수집 부족: {missing}")


def save_val_test(root: Path, val_per_class: int, test_per_class: int, quality: int):
    ds = load_dataset("ethz/food101", split="validation", streaming=True)
    counts = defaultdict(int)
    target = set(TARGET_CLASSES)
    per_class_total = val_per_class + test_per_class
    total_needed = per_class_total * len(target)
    saved = 0

    pbar = tqdm(total=total_needed, desc="val/test 저장")
    for sample in ds:
        label_name = ds.features["label"].int2str(sample["label"])
        if label_name not in target or counts[label_name] >= per_class_total:
            continue

        idx = counts[label_name]
        if idx < val_per_class:
            split = "val"
            split_idx = idx
        else:
            split = "test"
            split_idx = idx - val_per_class

        out = root / split / label_name / f"{label_name}_{split_idx:04d}.jpg"
        sample["image"].convert("RGB").save(out, quality=quality)
        counts[label_name] += 1
        saved += 1
        pbar.update(1)

        if saved >= total_needed:
            break
    pbar.close()

    missing = {c: per_class_total - counts[c] for c in target if counts[c] < per_class_total}
    if missing:
        raise RuntimeError(f"validation 데이터 수집 부족: {missing}")


def main():
    args = parse_args()
    ensure_dirs(args.output)
    print("대상 클래스:", ", ".join(TARGET_CLASSES))
    save_train(args.output, args.train_per_class, args.jpeg_quality)
    save_val_test(args.output, args.val_per_class, args.test_per_class, args.jpeg_quality)
    print(f"\n완료: {args.output.resolve()}")
    print(
        f"train={args.train_per_class * len(TARGET_CLASSES)}, "
        f"val={args.val_per_class * len(TARGET_CLASSES)}, "
        f"test={args.test_per_class * len(TARGET_CLASSES)}"
    )


if __name__ == "__main__":
    main()
