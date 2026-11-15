"""PyTorch dataset and transforms for the processed APTOS images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> T.Compose:
    common_tail = [
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if not train:
        return T.Compose([T.ToImage(), T.Resize((image_size, image_size)), *common_tail])
    # Fundus images have no canonical orientation, so flips and full rotations are safe.
    return T.Compose(
        [
            T.ToImage(),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            *common_tail,
        ]
    )


class FundusDataset(Dataset):
    """Loads all images into RAM once (APTOS at 224px is ~550 MB) for fast epochs."""

    def __init__(self, csv_path: Path, image_dir: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.labels = self.df.diagnosis.to_numpy(dtype=np.int64)
        self.images = [
            np.asarray(Image.open(image_dir / f"{i}.png").convert("RGB")) for i in self.df.id_code
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)
