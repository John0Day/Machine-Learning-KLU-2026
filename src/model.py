"""Baseline CNN model for GTSRB classification (Task 04).

Architecture (as specified in TASKS.md):
Input (3 x 32 x 32)
-> Conv(32, 3x3) + ReLU + MaxPool(2x2)
-> Conv(64, 3x3) + ReLU + MaxPool(2x2)
-> Flatten
-> Linear(256) + ReLU + Dropout(0.5)
-> Linear(43)

Note: Softmax is not applied in forward() because CrossEntropyLoss expects logits.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Simple baseline CNN for 43-class traffic sign classification."""

    def __init__(self, num_classes: int = 43, input_size: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
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
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            out = self.features(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
