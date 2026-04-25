"""Baseline CNN model for GTSRB classification (Task 04).

Architecture:
Input (3 x 32 x 32)
-> Conv(32, 3x3, padding=1) + BatchNorm + ReLU + MaxPool(2x2)  -> 32 x 16 x 16
-> Conv(64, 3x3, padding=1) + BatchNorm + ReLU + MaxPool(2x2)  -> 64 x 8 x 8
-> Conv(128, 3x3, padding=1) + BatchNorm + ReLU + MaxPool(2x2) -> 128 x 4 x 4
-> Flatten
-> Linear(256) + ReLU + Dropout(0.5)
-> Linear(43)

Note: Softmax is not applied in forward() because CrossEntropyLoss expects logits.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Baseline CNN for 43-class traffic sign classification.

    Improvements over the minimal baseline:
    - padding=1 preserves spatial dimensions before pooling
    - BatchNorm after each Conv stabilizes training (Lecture 4)
    - Third Conv block learns more abstract features (Lecture 5)
    - _infer_flatten_dim() keeps the classifier size-agnostic
    """

    def __init__(self, num_classes: int = 43, input_size: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 x 32 x 32 -> 32 x 16 x 16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32 x 16 x 16 -> 64 x 8 x 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64 x 8 x 8 -> 128 x 4 x 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        flatten_dim = self._infer_flatten_dim(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def _infer_flatten_dim(self, input_size: int) -> int:
        """Automatically compute the flattened size after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            out = self.features(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = BaselineCNN(num_classes=43, input_size=32)

    # Print a readable architecture summary for quick manual inspection.
    print(model)
    print()

    # Count trainable parameters to estimate model complexity and memory cost.
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Run a dummy forward pass to verify output tensor shape early.
    dummy = torch.zeros(8, 3, 32, 32)
    output = model(dummy)
    print(f"Input  shape: {dummy.shape}")
    print(f"Output shape: {output.shape}")   # expected: torch.Size([8, 43])
    assert output.shape == (8, 43), "Unexpected output shape."
    print("✓ Model OK")
