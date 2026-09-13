from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGE_SIZE


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    dataset_dir = Path(dataset_dir)

    train_ds = datasets.ImageFolder(dataset_dir / "train", transform=train_transform())
    val_ds = datasets.ImageFolder(dataset_dir / "val", transform=eval_transform())
    test_ds = datasets.ImageFolder(dataset_dir / "test", transform=eval_transform())

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise RuntimeError("train/val/test의 클래스 구성이 서로 다릅니다.")

    pin_memory = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_ds.classes
