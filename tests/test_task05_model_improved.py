"""Tests for Task 05 – Improved Models (model_improved.py).

This module verifies all four improved CNN variants:

    DeepCNN         – Deeper baseline with 4 Conv blocks and more filters.
                      Should have more parameters than the baseline.
    MobileNetTransfer – Pretrained MobileNetV2 with a custom classifier head.
                      Must expose a backbone/features attribute.
    LeakyReLUCNN    – Same as baseline but with LeakyReLU instead of ReLU.
                      Must contain at least one LeakyReLU module.
    StrideCNN       – Replaces MaxPool with strided convolutions (stride=2).
                      Must contain NO MaxPool2d layers (Lecture 5).

The tests cover:

- Output shape: every variant produces [batch_size, 43] logits.
- Numerical stability: no NaN or Inf in the forward pass.
- Gradient flow: gradients reach all layers during backpropagation.
- Architecture-specific constraints:
    * StrideCNN: no MaxPool2d layers (uses stride=2 Conv instead)
    * StrideCNN: accepts a `dropout` parameter for Optuna tuning
    * DeepCNN: more parameters than the baseline (deeper = more capacity)
    * LeakyReLUCNN: contains LeakyReLU activation modules
    * MobileNetTransfer: exposes a backbone or features attribute

Lecture reference: CNN architecture variants, learned vs. fixed downsampling
(Lecture 5), Dropout as regularization (Lecture 4).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import BaselineCNN
from model_improved import DeepCNN, LeakyReLUCNN, MobileNetTransfer, StrideCNN


# ── Helpers ────────────────────────────────────────────────────────────────

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Output shape (all variants) ────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass", [DeepCNN, LeakyReLUCNN, StrideCNN])
def test_output_shape(ModelClass):
    """Every improved model produces [batch_size, 43] logits."""
    model = ModelClass(num_classes=43, input_size=32)
    model.eval()
    x = torch.zeros(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 43), f"{ModelClass.__name__}: expected (8,43), got {out.shape}"


def test_mobilenet_output_shape():
    """MobileNetTransfer produces [batch_size, 43] logits."""
    model = MobileNetTransfer(num_classes=43)
    model.eval()
    x = torch.zeros(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 43), f"Expected (4,43), got {out.shape}"


# ── Numerical stability ────────────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass", [DeepCNN, LeakyReLUCNN, StrideCNN])
def test_no_nan_output(ModelClass):
    """Forward pass must not produce NaN values."""
    model = ModelClass(num_classes=43, input_size=32)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), f"{ModelClass.__name__} output contains NaN"


# ── Gradient flow ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("ModelClass", [DeepCNN, LeakyReLUCNN, StrideCNN])
def test_gradient_flow(ModelClass):
    """Gradients must reach all layers — no disconnected parameters."""
    model = ModelClass(num_classes=43, input_size=32)
    model.train()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    out.sum().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{ModelClass.__name__}: no gradient for {name}"


# ── StrideCNN specific ─────────────────────────────────────────────────────

def test_stride_cnn_no_maxpool():
    """StrideCNN must not use MaxPool2d — downsampling via stride=2 Conv instead.

    Using stride=2 in Conv layers is the architectural difference vs. the baseline.
    If MaxPool is still present, the 'learned downsampling' claim is false.
    """
    model = StrideCNN(num_classes=43, input_size=32)
    maxpool_layers = [m for m in model.modules() if isinstance(m, torch.nn.MaxPool2d)]
    assert len(maxpool_layers) == 0, (
        f"StrideCNN should have no MaxPool2d layers, found {len(maxpool_layers)}. "
        "Use stride=2 in Conv2d instead."
    )


def test_stride_cnn_accepts_dropout_param():
    """StrideCNN must accept a dropout parameter for Optuna hyperparameter tuning."""
    model = StrideCNN(num_classes=43, input_size=32, dropout=0.3)
    model.eval()
    x = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 43)


# ── DeepCNN specific ───────────────────────────────────────────────────────

def test_deep_cnn_more_params_than_baseline():
    """DeepCNN must have more parameters than BaselineCNN.

    A 'deeper' model with more blocks and filters should have higher capacity.
    If it has fewer parameters, the architecture is likely misconfigured.
    """
    baseline_params = _count_params(BaselineCNN(num_classes=43, input_size=32))
    deep_params = _count_params(DeepCNN(num_classes=43, input_size=32))
    assert deep_params > baseline_params, (
        f"DeepCNN ({deep_params:,}) should have more params than BaselineCNN ({baseline_params:,})"
    )


# ── LeakyReLUCNN specific ──────────────────────────────────────────────────

def test_leakyrelu_cnn_has_leaky_relu():
    """LeakyReLUCNN must contain LeakyReLU activations.

    The 'dead neuron' fix only works if LeakyReLU actually replaces ReLU.
    If only standard ReLU is present, the architectural change is missing.
    """
    model = LeakyReLUCNN(num_classes=43, input_size=32)
    leaky_layers = [m for m in model.modules() if isinstance(m, torch.nn.LeakyReLU)]
    assert len(leaky_layers) > 0, "LeakyReLUCNN contains no LeakyReLU modules"


# ── MobileNetTransfer specific ─────────────────────────────────────────────

def test_mobilenet_has_backbone():
    """MobileNetTransfer must expose a backbone or features attribute."""
    model = MobileNetTransfer(num_classes=43)
    has_backbone = hasattr(model, "backbone") or hasattr(model, "features")
    assert has_backbone, (
        "MobileNetTransfer should have a 'backbone' or 'features' attribute "
        "for the pretrained MobileNetV2 feature extractor."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
