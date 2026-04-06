"""GTSRB data preprocessing, augmentation, and DataLoader creation.

Reads the local GTSRB training images (downloaded via fetch_gtsrb.sh) and
produces three ready-to-use PyTorch DataLoaders: train, validation, and test.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --data-root data/raw --img-size 32 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GTSRBDataset(Dataset):
    """PyTorch Dataset for the local GTSRB training images.

    Reads .ppm images and their class labels from the official CSV annotation
    files that ship with the GTSRB download.

    Args:
        images_dir: Path to the ``Final_Training/Images`` folder.
        transform:  Optional torchvision transform applied to each image.
    """

    def __init__(self, images_dir: Path, transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        class_dirs = sorted(
            d for d in images_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
        if not class_dirs:
            raise RuntimeError(f"No class folders found in {images_dir}")

        for class_dir in class_dirs:
            class_id = int(class_dir.name)
            gt_file = class_dir / f"GT-{class_dir.name}.csv"
            if not gt_file.exists():
                raise FileNotFoundError(f"Missing annotation file: {gt_file}")

            with gt_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    image_path = class_dir / row["Filename"]
                    self.samples.append((image_path, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet mean & std – good starting point for GTSRB
_MEAN = (0.3337, 0.3064, 0.3171)
_STD  = (0.2672, 0.2564, 0.2629)


def get_train_transform(img_size: int = 32) -> transforms.Compose:
    """Augmented transform for the training split."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_val_test_transform(img_size: int = 32) -> transforms.Compose:
    """Deterministic transform for validation and test splits (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: Path = Path("data/raw"),
    img_size: int = 32,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from local GTSRB files.

    The full training set is split into train / val / test using the given
    fractions. Augmentation is applied only to the training split.

    Args:
        data_root:      Root folder containing ``Final_Training/Images``.
        img_size:       Target image size (square) in pixels.
        batch_size:     Number of samples per batch.
        val_fraction:   Fraction of data used for validation.
        test_fraction:  Fraction of data used for testing.
        num_workers:    DataLoader worker processes.
        seed:           Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    images_dir = _resolve_images_dir(data_root)

    # Load full dataset without transform first to get indices for splitting
    full_dataset = GTSRBDataset(images_dir, transform=None)
    total = len(full_dataset)

    n_test = int(total * test_fraction)
    n_val  = int(total * val_fraction)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Wrap subsets with the correct transforms
    train_dataset = _TransformedSubset(train_subset, get_train_transform(img_size))
    val_dataset   = _TransformedSubset(val_subset,   get_val_test_transform(img_size))
    test_dataset  = _TransformedSubset(test_subset,  get_val_test_transform(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


class _TransformedSubset(Dataset):
    """Wraps a Subset and applies a transform on top."""

    def __init__(self, subset, transform: Callable) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_images_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "Final_Training" / "Images",
        data_root / "GTSRB_Final_Training_Images" / "Final_Training" / "Images",
        data_root,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        f"Could not find GTSRB image directory under {data_root}. "
        "Run scripts/fetch_gtsrb.sh first."
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: Path,
) -> None:
    """Print split sizes, check batch shapes, and save a sample grid."""
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    total   = n_train + n_val + n_test

    print(f"Total samples : {total}")
    print(f"Train samples : {n_train} ({100 * n_train / total:.1f} %)")
    print(f"Val   samples : {n_val}   ({100 * n_val   / total:.1f} %)")
    print(f"Test  samples : {n_test}  ({100 * n_test  / total:.1f} %)")

    images, labels = next(iter(train_loader))
    print(f"\nBatch image shape : {images.shape}")   # e.g. torch.Size([64, 3, 32, 32])
    print(f"Batch label shape : {labels.shape}")    # e.g. torch.Size([64])
    print(f"Image dtype       : {images.dtype}")
    print(f"Label dtype       : {labels.dtype}")
    print(f"Pixel min/max     : {images.min():.3f} / {images.max():.3f}")

    # Save a small sample grid to results/
    results_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_grid(images[:16], labels[:16], results_dir / "preprocessing_sample_grid.png")
    print(f"\nSample grid saved → {results_dir / 'preprocessing_sample_grid.png'}")
    print("\n✓ Preprocessing OK")


def _save_sample_grid(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Denormalise and save a 4×4 grid of sample images."""
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    imgs = (images * std + mean).clamp(0, 1)

    n = min(16, len(imgs))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
            ax.set_title(f"Class {labels[i].item()}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample Training Batch (after augmentation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GTSRB DataLoaders")
    parser.add_argument("--data-root",   type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--img-size",    type=int,  default=32)
    parser.add_argument("--batch-size",  type=int,  default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Building DataLoaders …")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    _verify(train_loader, val_loader, test_loader, args.results_dir)
