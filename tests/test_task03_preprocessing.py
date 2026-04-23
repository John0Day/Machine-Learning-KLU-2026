"""Tests for Task 03 – Data Preprocessing (preprocessing.py).

This test module verifies the image preprocessing and data augmentation pipeline.
Preprocessing is a critical step that directly affects model performance — wrong
normalization or augmentation can silently degrade accuracy without obvious errors.

The tests cover:

- Output shape: both train and val/test transforms resize images to [3, 32, 32]
- Normalization: pixel values are normalized using GTSRB mean/std (not raw 0-1)
- Augmentation (train): the train transform must be stochastic (random rotations,
  color jitter, affine transforms) — same image should produce different tensors
- Determinism (val/test): the val/test transform must always produce the same output
  for the same input — no random augmentation allowed during evaluation
- Scalability: transforms also work correctly with img_size=64


"""

import sys
from pathlib import Path

import pytest
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import (
    get_train_transform,
    get_val_test_transform,
)


# ── Task 03: Transforms ────────────────────────────────────────────────────

def test_val_test_transform_output_shape():
    """Val/test transform produces [3, 32, 32] tensors."""
    from PIL import Image
    import numpy as np

    transform = get_val_test_transform(img_size=32)
    dummy_img = Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype("uint8")
    )
    tensor = transform(dummy_img)
    assert tensor.shape == (3, 32, 32), f"Expected (3,32,32), got {tensor.shape}"


def test_train_transform_output_shape():
    """Train transform produces [3, 32, 32] tensors."""
    from PIL import Image
    import numpy as np

    transform = get_train_transform(img_size=32)
    dummy_img = Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype("uint8")
    )
    tensor = transform(dummy_img)
    assert tensor.shape == (3, 32, 32), f"Expected (3,32,32), got {tensor.shape}"


def test_val_test_transform_normalized():
    """Val/test transform normalizes pixel values (not raw 0-1)."""
    from PIL import Image
    import numpy as np

    transform = get_val_test_transform(img_size=32)
    # White image — after normalization values should differ from 1.0
    white_img = Image.fromarray(
        (np.ones((64, 64, 3)) * 255).astype("uint8")
    )
    tensor = transform(white_img)
    # Mean of GTSRB is ~0.33, std ~0.27 → normalized white pixel ≈ 2.5
    assert tensor.max().item() > 1.5, "Normalization seems missing"


def test_train_transform_is_stochastic():
    """Train transform should produce different results on the same image."""
    from PIL import Image
    import numpy as np

    transform = get_train_transform(img_size=32)
    dummy_img = Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype("uint8")
    )
    results = [transform(dummy_img) for _ in range(5)]
    # At least one pair should differ (augmentation is random)
    all_equal = all(torch.equal(results[0], r) for r in results[1:])
    assert not all_equal, "Train transform appears deterministic — augmentation may be missing"


def test_val_test_transform_is_deterministic():
    """Val/test transform must be deterministic (no augmentation)."""
    from PIL import Image
    import numpy as np

    transform = get_val_test_transform(img_size=32)
    dummy_img = Image.fromarray(
        (np.random.rand(64, 64, 3) * 255).astype("uint8")
    )
    t1 = transform(dummy_img)
    t2 = transform(dummy_img)
    assert torch.equal(t1, t2), "Val/test transform is not deterministic"


def test_transforms_64px():
    """Transforms work with img_size=64 as well."""
    from PIL import Image
    import numpy as np

    for transform in [get_train_transform(64), get_val_test_transform(64)]:
        img = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype("uint8"))
        tensor = transform(img)
        assert tensor.shape == (3, 64, 64)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
