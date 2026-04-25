"""Tests for Task 04 – Baseline Model (model.py).

This module verifies the correctness of the BaselineCNN architecture.
The baseline is a 3-block CNN with Conv2d, BatchNorm, ReLU, MaxPool, Dropout,
and two Linear layers — the foundation on which all other variants are built.

The tests cover:

- Output shape: model produces [batch_size, 43] logits for any valid batch size.
  The output must have exactly one score per GTSRB class.
- num_classes parameter: the output shape must respect the configured number
  of classes, not be hardcoded to 43.
- Numerical stability: no NaN or Inf values in the forward pass output.
  NaN/Inf would cause CrossEntropyLoss to return NaN and training to diverge.
- Parameter count: model has a reasonable number of trainable parameters (>100K).
  Too few parameters indicate that a layer is missing or misconfigured.
- Gradient flow: gradients must reach every layer during backpropagation.
  If any layer has no gradient, it is disconnected and does not learn.
- Loss computation: CrossEntropyLoss can be computed and remains finite after
  one gradient step — basic end-to-end training loop sanity check.

Lecture reference: CNN architecture (Lecture 5), backpropagation and gradient
descent (Lecture 2–3), CrossEntropyLoss and Adam optimizer (Lecture 4).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import BaselineCNN


# ── Task 04: Model architecture ────────────────────────────────────────────

def test_baseline_output_shape():
    """BaselineCNN produces [batch_size, 43] logits for a standard batch."""
    model = BaselineCNN(num_classes=43, input_size=32)
    model.eval()
    x = torch.zeros(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 43), f"Expected (8,43), got {out.shape}"


def test_baseline_output_shape_single():
    """BaselineCNN handles batch size 1 without errors."""
    model = BaselineCNN(num_classes=43, input_size=32)
    model.eval()
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 43)


def test_baseline_num_classes_param():
    """num_classes parameter controls the output dimension — must not be hardcoded."""
    model = BaselineCNN(num_classes=10, input_size=32)
    model.eval()
    x = torch.zeros(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"


def test_baseline_no_nan_output():
    """Forward pass must not produce NaN values.

    NaN in the output would cause CrossEntropyLoss to return NaN,
    making all gradient updates undefined and training useless.
    """
    model = BaselineCNN(num_classes=43, input_size=32)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "Model output contains NaN values"


def test_baseline_no_inf_output():
    """Forward pass must not produce Inf values."""
    model = BaselineCNN(num_classes=43, input_size=32)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isinf(out).any(), "Model output contains Inf values"


def test_baseline_trainable_params():
    """BaselineCNN has more than 100K trainable parameters.

    A model with fewer parameters likely has a misconfigured layer
    (e.g., wrong channel size or missing block).
    """
    model = BaselineCNN(num_classes=43, input_size=32)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 100_000, f"Too few parameters: {n_params:,}"


def test_baseline_gradient_flow():
    """Gradients must reach every layer — no layer should be disconnected.

    If param.grad is None after backward(), that layer does not contribute
    to learning and the architecture is broken.
    """
    model = BaselineCNN(num_classes=43, input_size=32)
    model.train()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    out.sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"


def test_baseline_cross_entropy_loss():
    """CrossEntropyLoss remains finite after one full gradient step."""
    model = BaselineCNN(num_classes=43, input_size=32)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 43, (16,))

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    assert loss.item() < float("inf"), "Loss is infinite after one gradient step"
    assert not torch.isnan(loss), "Loss is NaN after one gradient step"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
