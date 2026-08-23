"""Dataset utilities for UCS-Expert training and evaluation."""

from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CoralDataLoader(Dataset):
    """Load paired coral images and binary masks.

    A dataset root may contain ``train/images`` and ``test/images`` or may
    point directly to a split containing ``images`` and ``masks``.
    """

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        resize_size: Sequence[int] = (512, 512),
        label_resize_size: Sequence[int] = (),
        train_ratio: float = 1.0,
    ) -> None:
        if not 0 < train_ratio <= 1:
            raise ValueError(
                f"train_ratio must be in (0, 1], got {train_ratio}")
        self.train = train
        self.resize_size = _validate_size(resize_size)
        self.label_resize_size = (_validate_size(label_resize_size)
                                  if label_resize_size else self.resize_size)
        self.class_num = 1
        self.image_size = self.resize_size[0]

        split_root = _resolve_split_root(Path(data_root), train)
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"
        image_paths = sorted(path for path in image_dir.iterdir()
                             if path.suffix.lower() in IMAGE_EXTENSIONS)
        keep = int(len(image_paths) * train_ratio)
        image_paths = image_paths[:keep]

        if not image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

        self.samples = []
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}: {mask_path}")
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Unable to read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           self.resize_size[::-1],
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask,
                          self.label_resize_size[::-1],
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        if not mask.any():
            raise ValueError(
                f"Mask contains no foreground pixels: {mask_path}")
        if self.train:
            image, mask = random_flip(image, mask)

        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())

        if self.train:
            height, width = mask.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(width - 1, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(height - 1, y_max + np.random.randint(0, 20))

        image_tensor = torch.from_numpy(
            image.astype(np.float32).transpose(2, 0, 1) / 255.0)
        mask_tensor = torch.from_numpy(mask[None]).long()
        box_tensor = torch.tensor([x_min, y_min, x_max, y_max],
                                  dtype=torch.float32)
        class_tensor = torch.tensor(0, dtype=torch.long)
        return (
            image_tensor,
            mask_tensor,
            box_tensor,
            class_tensor,
            str(image_path),
            str(mask_path),
        )


def _resolve_split_root(data_root: Path, train: bool) -> Path:
    split = "train" if train else "test"
    split_root = data_root / split
    if (split_root / "images").is_dir() and (split_root / "masks").is_dir():
        return split_root
    if (data_root / "images").is_dir() and (data_root / "masks").is_dir():
        return data_root
    raise FileNotFoundError(
        f"Expected {split_root}/{{images,masks}} or {data_root}/{{images,masks}}"
    )


def random_flip(image: np.ndarray,
                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.random() < 0.5:
        image, mask = np.fliplr(image), np.fliplr(mask)
    if np.random.random() < 0.5:
        image, mask = np.flipud(image), np.flipud(mask)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _validate_size(size: Sequence[int]) -> Tuple[int, int]:
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError(
            f"Expected a positive (height, width) size, got {size}")
    return int(size[0]), int(size[1])
