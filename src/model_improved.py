"""Improved models for GTSRB classification (Task 05).

Variants provided for comparison against the baseline:

    Variant A – DeepCNN:
        Deeper version of the baseline with 4 Conv blocks and more filters.

    Variant B – MobileNetV2 (Transfer Learning):
        Pretrained MobileNetV2 with a custom classifier head for 43 classes.

    Variant C – LeakyReLU CNN:
        Same as baseline but uses Leaky ReLU instead of ReLU.
        Addresses the "dead neuron" problem described in Lecture 4.

    Variant D – StrideCNN:
        Replaces MaxPool with strided convolutions (stride=2).
        Tests whether learned downsampling outperforms fixed MaxPool (Lecture 5).

Usage:
    from src.model_improved import DeepCNN, MobileNetTransfer, LeakyReLUCNN, StrideCNN
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


# ---------------------------------------------------------------------------
# Variant A – Deeper CNN
# ---------------------------------------------------------------------------

class DeepCNN(nn.Module):
    """Deeper CNN with 4 convolutional blocks.

    Architecture:
        Input (3 x 32 x 32)
        -> Conv(32)  + BN + ReLU + MaxPool  -> 32 x 16 x 16
        -> Conv(64)  + BN + ReLU + MaxPool  -> 64 x 8 x 8
        -> Conv(128) + BN + ReLU + MaxPool  -> 128 x 4 x 4
        -> Conv(256) + BN + ReLU + MaxPool  -> 256 x 2 x 2
        -> Flatten -> Linear(512) + ReLU + Dropout -> Linear(43)
    """

    def __init__(self, num_classes: int = 43, input_size: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 32 x 32 → 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # Normalize activations → stable gradients (Lecture 4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 x 16 x 16 → 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 x 8 x 8 → 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 x 4 x 4 → 256 x 2 x 2  ← extra block vs. baseline
            # More filters allow the network to learn higher-level combinations
            # of features (Lecture 5: deeper networks capture more abstract patterns).
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flatten_dim = self._infer_flatten_dim(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 512),  # Larger hidden layer to match deeper feature extractor
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # Regularization: prevents overfitting (Lecture 4)
            nn.Linear(512, num_classes),
        )

    def _infer_flatten_dim(self, input_size: int) -> int:
        """Dynamically compute the flattened size after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            out = self.features(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features, flatten, and classify. Returns raw logits."""
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Variant B – MobileNetV2 Transfer Learning
# ---------------------------------------------------------------------------

class MobileNetTransfer(nn.Module):
    """MobileNetV2 pretrained on ImageNet with a custom classifier for GTSRB.

    Transfer learning strategy:
        - freeze_backbone=True:  Only the classifier head is trained.
                                  Fast, good when data is limited.
        - freeze_backbone=False: All weights are fine-tuned (full fine-tuning).
                                  Slower, usually better accuracy.

    Args:
        num_classes:     Number of output classes (43 for GTSRB).
        freeze_backbone: If True, freeze all MobileNet layers except the head.
        input_size:      Expected input image size. MobileNet needs at least 32x32.
    """

    def __init__(
        self,
        num_classes: int = 43,
        freeze_backbone: bool = False,
        input_size: int = 32,
    ) -> None:
        super().__init__()

        # Load pretrained MobileNetV2
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Keep the feature extractor
        self.features = backbone.features

        # Replace the classifier head with one suited for 43 classes
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run pretrained feature extractor, global pool, and custom classifier."""
        x = self.features(x)
        x = self.pool(x)          # Global average pooling → [batch, 1280, 1, 1]
        x = torch.flatten(x, 1)  # Flatten → [batch, 1280]
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Variant C – LeakyReLU CNN (Lecture 4: dead neuron problem)
# ---------------------------------------------------------------------------

class LeakyReLUCNN(nn.Module):
    """Same as baseline but uses Leaky ReLU instead of ReLU.

    Leaky ReLU: φ(z) = z if z > 0, else 0.01 * z
    This prevents dead neurons where ReLU outputs zero for all inputs,
    causing the gradient to be zero and the neuron to stop learning.
    """

    def __init__(self, num_classes: int = 43, input_size: int = 32, negative_slope: float = 0.01) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 32 x 32 → 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope, inplace=True),  # φ(z) = z if z>0, else 0.01·z
            nn.MaxPool2d(2),

            # Block 2: 32 x 16 x 16 → 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 x 8 x 8 → 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.MaxPool2d(2),
        )

        flatten_dim = self._infer_flatten_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(p=0.5),   # Regularization: prevents overfitting (Lecture 4)
            nn.Linear(256, num_classes),
        )

    def _infer_flatten_dim(self, input_size: int) -> int:
        """Dynamically compute the flattened size after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features, flatten, and classify. Returns raw logits."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Variant D – StrideCNN: strided Conv instead of MaxPool (Lecture 5)
# ---------------------------------------------------------------------------

class StrideCNN(nn.Module):
    """Replaces MaxPool with strided convolutions (stride=2).

    Lecture 5 explains striding as an alternative to pooling for
    downsampling. Unlike MaxPool which discards 75% of values by
    taking the max, strided convolutions learn how to downsample,
    potentially preserving more useful information.
    """

    def __init__(
        self,
        num_classes: int = 43,
        input_size: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # stride=2 replaces MaxPool → learns downsampling
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),  # ← stride instead of MaxPool

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
        )

        flatten_dim = self._infer_flatten_dim(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def _infer_flatten_dim(self, input_size: int) -> int:
        """Dynamically compute the flattened size after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features, flatten, and classify. Returns raw logits."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy = torch.zeros(8, 3, 32, 32)
    models_to_test = [
        ("Variant A – DeepCNN",       DeepCNN(num_classes=43, input_size=32)),
        ("Variant B – MobileNetV2",   MobileNetTransfer(num_classes=43, input_size=32)),
        ("Variant C – LeakyReLU CNN", LeakyReLUCNN(num_classes=43, input_size=32)),
        ("Variant D – StrideCNN",     StrideCNN(num_classes=43, input_size=32)),
    ]

    print(f"\n{'='*55}")
    print(f"{'Model':<30} {'Params':>12} {'Output'}")
    print(f"{'='*55}")

    for name, model in models_to_test:
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert out.shape == (8, 43), f"Shape error in {name}"
        print(f"{name:<30} {params:>12,} {out.shape}")

    print(f"{'='*55}")
    print("✓ All models OK")
