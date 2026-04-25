"""Tests for Task 06 – Evaluation (evaluate.py).

This module verifies the evaluation pipeline used to assess trained models
on the GTSRB test set. The evaluation goes beyond simple accuracy and includes
interpretability (Grad-CAM), robustness testing, and per-class analysis.

The tests cover:

- Normalize/Denormalize roundtrip: denormalize(normalize(x)) ≈ x for images
  in [0, 1]. If this fails, Grad-CAM visualizations will show wrong colours.
- Denormalize clamping: output must always stay in [0, 1] even for extreme inputs.

- Perturbations (robustness testing):
    apply_noise: Gaussian noise must preserve tensor shape and change pixel values.
    apply_blur: Gaussian blur must preserve tensor shape and change pixel values.
    These simulate real-world degradations like sensor noise and motion blur.

- Top-5 Accuracy: compute_top5_accuracy returns a float in [0, 1].
  Top-5 means the correct class appears in the model's top 5 predictions.

- Per-class accuracy plot: save_per_class_accuracy_plot creates a PNG file
  and returns a numpy array of length 43 (one accuracy per GTSRB class).

- Precision/Recall curve: save_precision_recall_curve creates a PNG file
  showing precision and recall per class side by side.

- Grad-CAM: GradCAM.generate() produces a 2D heatmap of shape [32, 32]
  with values normalized to [0, 1]. find_last_conv_layer() correctly
  identifies the last Conv2d layer for gradient hooking.

Lecture reference: evaluation metrics (Lecture 4), Grad-CAM interpretability
and CNN feature visualization (Lecture 5).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluate import (
    GradCAM,
    apply_blur,
    apply_noise,
    compute_top5_accuracy,
    denormalize,
    find_last_conv_layer,
    normalize,
    save_per_class_accuracy_plot,
    save_precision_recall_curve,
)
from model import BaselineCNN


# ── Helpers ────────────────────────────────────────────────────────────────

def _dummy_model():
    """Small BaselineCNN in eval mode for fast testing."""
    model = BaselineCNN(num_classes=43, input_size=32)
    model.eval()
    return model


def _dummy_loader(n_batches=3, batch_size=8):
    """DataLoader with random images and labels for 43-class GTSRB."""
    dataset = torch.utils.data.TensorDataset(
        torch.randn(n_batches * batch_size, 3, 32, 32),
        torch.randint(0, 43, (n_batches * batch_size,)),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# ── Normalize / Denormalize ────────────────────────────────────────────────

def test_normalize_denormalize_roundtrip():
    """denormalize(normalize(x)) ≈ x for images in [0, 1].

    If this roundtrip fails, Grad-CAM heatmaps will be overlaid on
    incorrectly coloured images in the final report visualizations.
    """
    x = torch.rand(4, 3, 32, 32)
    reconstructed = denormalize(normalize(x))
    assert torch.allclose(x, reconstructed, atol=1e-5), (
        "Normalize/denormalize roundtrip failed — "
        "check that denormalize uses the same mean/std as normalize."
    )


def test_denormalize_clamps_to_01():
    """denormalize output must stay in [0, 1] even for extreme inputs.

    Unclamped values would cause matplotlib to clip silently and produce
    wrong colours in saved visualizations.
    """
    x = torch.randn(4, 3, 32, 32) * 10  # extreme values outside normal range
    out = denormalize(x)
    assert out.min().item() >= 0.0, f"Denormalize output below 0: {out.min().item()}"
    assert out.max().item() <= 1.0, f"Denormalize output above 1: {out.max().item()}"


# ── Perturbations ──────────────────────────────────────────────────────────

def test_apply_noise_preserves_shape():
    """apply_noise must preserve the input tensor shape."""
    x = torch.randn(4, 3, 32, 32)
    noisy = apply_noise(x, std=0.1)
    assert noisy.shape == x.shape, f"Shape changed: {x.shape} → {noisy.shape}"


def test_apply_noise_changes_values():
    """apply_noise must actually modify pixel values, not return the input unchanged."""
    x = torch.zeros(4, 3, 32, 32)
    noisy = apply_noise(x, std=0.1)
    assert not torch.equal(x, noisy), "apply_noise returned the unchanged input"


def test_apply_blur_preserves_shape():
    """apply_blur must preserve the input tensor shape."""
    x = torch.randn(4, 3, 32, 32)
    blurred = apply_blur(x, kernel_size=5)
    assert blurred.shape == x.shape, f"Shape changed: {x.shape} → {blurred.shape}"


def test_apply_blur_changes_values():
    """apply_blur must actually smooth pixel values, not return the input unchanged."""
    x = torch.randn(4, 3, 32, 32)
    blurred = apply_blur(x, kernel_size=5)
    assert not torch.equal(x, blurred), "apply_blur returned the unchanged input"


# ── Top-5 Accuracy ─────────────────────────────────────────────────────────

def test_top5_accuracy_in_range():
    """compute_top5_accuracy must return a float in [0, 1].

    Top-5 accuracy ≥ Top-1 accuracy. For a 43-class problem a random model
    achieves ~11.6% Top-1 but ~40% Top-5.
    """
    model = _dummy_model()
    loader = _dummy_loader()
    acc = compute_top5_accuracy(model, loader, torch.device("cpu"), max_batches=0)
    assert 0.0 <= acc <= 1.0, f"Top-5 accuracy out of range: {acc}"


# ── Per-class accuracy plot ────────────────────────────────────────────────

def test_per_class_accuracy_plot_creates_file(tmp_path):
    """save_per_class_accuracy_plot must create a PNG and return 43 accuracies."""
    true_labels = torch.randint(0, 43, (200,))
    pred_labels = torch.randint(0, 43, (200,))
    class_names = [f"Class {i}" for i in range(43)]
    out_path = tmp_path / "per_class.png"

    result = save_per_class_accuracy_plot(true_labels, pred_labels, class_names, out_path)

    assert out_path.exists(), "PNG file was not created"
    assert isinstance(result, np.ndarray), "Return value must be a numpy array"
    assert len(result) == 43, f"Expected 43 accuracies, got {len(result)}"


# ── Precision/Recall curve ─────────────────────────────────────────────────

def test_precision_recall_curve_creates_file(tmp_path):
    """save_precision_recall_curve must create a PNG file."""
    true_labels = torch.randint(0, 43, (200,))
    pred_labels = torch.randint(0, 43, (200,))
    class_names = [f"Class {i}" for i in range(43)]
    out_path = tmp_path / "pr_curve.png"

    save_precision_recall_curve(true_labels, pred_labels, class_names, out_path)

    assert out_path.exists(), "Precision/Recall PNG file was not created"


# ── Grad-CAM ───────────────────────────────────────────────────────────────

def test_gradcam_output_shape():
    """GradCAM.generate() must produce a 2D heatmap matching the input spatial size.

    The heatmap is overlaid on the input image, so shape mismatch would cause
    a crash or misaligned visualization in the final report.
    """
    model = _dummy_model()
    model.train()  # gradients required for hook registration
    target_layer = find_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 32, 32)
    cam = gradcam.generate(x, class_idx=0)
    assert cam.shape == (32, 32), f"Expected (32,32) heatmap, got {cam.shape}"


def test_gradcam_values_in_01():
    """GradCAM heatmap values must be normalized to [0, 1].

    Values outside [0, 1] would cause matplotlib to clip the colormap
    and produce misleading visualizations of which regions matter most.
    """
    model = _dummy_model()
    model.train()
    target_layer = find_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 32, 32)
    cam = gradcam.generate(x, class_idx=5)
    assert cam.min().item() >= 0.0, f"Heatmap min below 0: {cam.min().item()}"
    assert cam.max().item() <= 1.0, f"Heatmap max above 1: {cam.max().item()}"


def test_find_last_conv_layer():
    """find_last_conv_layer must return a Conv2d module for gradient hooking."""
    model = _dummy_model()
    layer = find_last_conv_layer(model)
    assert isinstance(layer, torch.nn.Conv2d), (
        f"Expected Conv2d, got {type(layer).__name__}. "
        "Grad-CAM hooks require a convolutional layer."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
