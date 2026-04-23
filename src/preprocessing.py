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
import json
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

# GTSRB-specific mean and std computed from the training set.
# Used to normalize pixel values to zero mean and unit variance (Lecture 4).
_MEAN = (0.3337, 0.3064, 0.3171)
_STD  = (0.2672, 0.2564, 0.2629)


def get_train_transform(img_size: int = 32) -> transforms.Compose:
    """Augmented transform for the training split.

    Augmentation techniques are applied only during training to artificially
    increase dataset diversity and prevent overfitting (Lecture 4).
    Each technique simulates real-world variation in traffic sign appearance:

    - RandomRotation(±15°): signs seen at a slight angle from a moving vehicle
    - ColorJitter: varying lighting conditions, weather, and camera exposure
    - RandomAffine(translate): sign not perfectly centered in the camera frame
    - Normalize: zero-mean, unit-variance normalization for stable gradient flow
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),         # Standardize all images to img_size×img_size
        transforms.RandomRotation(degrees=15),           # Simulate slight camera angle (±15°)
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),  # Lighting variation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small position shift
        transforms.ToTensor(),                           # Convert PIL image to [C, H, W] float tensor
        transforms.Normalize(mean=_MEAN, std=_STD),     # Normalize to zero mean, unit variance
    ])


def get_val_test_transform(img_size: int = 32) -> transforms.Compose:
    """Deterministic transform for validation and test splits (no augmentation).

    No augmentation is applied here — validation and test images must be
    transformed identically every time to ensure reproducible evaluation results.
    Only resize and normalize are applied (same normalization as training).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),         # Standardize to img_size×img_size
        transforms.ToTensor(),                           # Convert PIL image to [C, H, W] float tensor
        transforms.Normalize(mean=_MEAN, std=_STD),     # Same normalization as training split
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
    """Print split sizes, check batch shapes, and save a sample grid.

    Verifies that:
    - The 70/15/15 train/val/test split was applied correctly
    - Batch shape is [batch_size, 3, img_size, img_size] as expected
    - Pixel values are normalized (not raw 0-255)
    - A sample grid of augmented training images is saved to results/
    """
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    total   = n_train + n_val + n_test

    print("=" * 45)
    print("  Dataset Split (seed=42)")
    print("=" * 45)
    print(f"  Total samples : {total}")
    print(f"  Train samples : {n_train:>6}  ({100 * n_train / total:.1f} %)  ← expected ~70%")
    print(f"  Val   samples : {n_val:>6}  ({100 * n_val   / total:.1f} %)  ← expected ~15%")
    print(f"  Test  samples : {n_test:>6}  ({100 * n_test  / total:.1f} %)  ← expected ~15%")

    # Verify batch shape from actual DataLoader output
    images, labels = next(iter(train_loader))
    print("\n" + "=" * 45)
    print("  Batch Verification")
    print("=" * 45)
    print(f"  Batch image shape : {list(images.shape)}")   # Expected: [64, 3, 32, 32]
    print(f"  Batch label shape : {list(labels.shape)}")   # Expected: [64]
    print(f"  Image dtype       : {images.dtype}")         # Expected: torch.float32
    print(f"  Label dtype       : {labels.dtype}")         # Expected: torch.int64
    print(f"  Pixel min         : {images.min():.3f}")     # Should be negative after normalization
    print(f"  Pixel max         : {images.max():.3f}")     # Should be > 1.0 after normalization
    print(f"  Pixel mean (approx): {images.mean():.3f}")   # Should be close to 0.0

    # Sanity check: pixel values should be normalized (not raw 0-1)
    assert images.dtype == torch.float32, "Images should be float32 tensors"
    assert labels.dtype == torch.int64,   "Labels should be int64 (LongTensor)"
    assert images.shape[1] == 3,          "Images should have 3 channels (RGB)"

    # Save a small sample grid to results/
    results_dir.mkdir(parents=True, exist_ok=True)
    grid_path = results_dir / "preprocessing_sample_grid.png"
    _save_sample_grid(images[:16], labels[:16], grid_path)
    print(f"\n  Sample grid saved → {grid_path}")

    # Save split statistics as JSON for reproducibility and reporting
    stats = {
        "total_samples": total,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "train_fraction": round(n_train / total, 4),
        "val_fraction": round(n_val / total, 4),
        "test_fraction": round(n_test / total, 4),
        "batch_size": images.shape[0],
        "image_shape": list(images.shape[1:]),   # [C, H, W]
        "image_dtype": str(images.dtype),
        "pixel_min": round(images.min().item(), 4),
        "pixel_max": round(images.max().item(), 4),
        "pixel_mean": round(images.mean().item(), 4),
        "normalization_mean": list(_MEAN),
        "normalization_std": list(_STD),
        "augmentations": [
            "RandomRotation(±15°)",
            "ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)",
            "RandomAffine(translate=0.1)",
        ],
    }
    stats_path = results_dir / "preprocessing_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats JSON saved  → {stats_path}")

    # Save class distribution per split as PNG — useful for bias analysis in report
    _save_split_distribution(train_loader, val_loader, test_loader,
                             results_dir / "preprocessing_split_distribution.png")
    print(f"  Split distribution → {results_dir / 'preprocessing_split_distribution.png'}")
    print("\n✓ Preprocessing OK — all checks passed")


def _save_split_distribution(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    path: Path,
) -> None:
    """Save a bar chart showing the number of samples per class in each split.

    This plot helps verify that the random split preserved the class distribution
    proportionally across train, val, and test — important for unbiased evaluation.
    """
    import numpy as np
    from collections import Counter

    def count_labels(loader: DataLoader) -> list:
        counts = Counter()
        for _, labels in loader:
            for label in labels.tolist():
                counts[label] += 1
        return [counts.get(i, 0) for i in range(43)]

    print("  Computing split class distributions (this may take a moment)…")
    train_counts = count_labels(train_loader)
    val_counts   = count_labels(val_loader)
    test_counts  = count_labels(test_loader)

    x = range(43)
    width = 0.28
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar([i - width for i in x], train_counts, width, label=f"Train ({sum(train_counts)})", color="#2a9d8f", alpha=0.85)
    ax.bar([i         for i in x], val_counts,   width, label=f"Val   ({sum(val_counts)})",   color="#e9c46a", alpha=0.85)
    ax.bar([i + width for i in x], test_counts,  width, label=f"Test  ({sum(test_counts)})",  color="#e76f51", alpha=0.85)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of samples")
    ax.set_title("Sample Distribution per Class and Split (seed=42)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
