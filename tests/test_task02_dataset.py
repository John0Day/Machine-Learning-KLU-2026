"""Tests for Task 02 – Dataset Loading (dataset.py).

This module verifies the correctness of the GTSRB dataset metadata and the
class mapping utility. The dataset loader is the foundation of the entire
project — if class labels are wrong or missing, every downstream model trains
on incorrect targets and all accuracy numbers become meaningless.

The tests cover:

- SIGN_LABELS count: exactly 43 labels must be defined (one per GTSRB class)
- No duplicates: each label name must be unique so classes are distinguishable
- No empty strings: every class must have a non-empty human-readable name
- Known class names: spot-check a few well-known classes to catch accidental
  label shuffling or off-by-one errors in the class index
- save_class_mapping: writes a CSV file that contains all 43 class IDs

Lecture reference: supervised learning requires correct label assignment
(Lecture 2). Any labelling error directly corrupts the training signal.
"""

import csv
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import SIGN_LABELS, save_class_mapping


# ── Task 02: Dataset labels ────────────────────────────────────────────────

def test_sign_labels_count():
    """GTSRB has exactly 43 traffic sign classes."""
    assert len(SIGN_LABELS) == 43, (
        f"Expected 43 labels, got {len(SIGN_LABELS)}. "
        "Check that no class was accidentally added or removed."
    )


def test_sign_labels_no_duplicates():
    """Each class label must be unique — duplicates would make classes indistinguishable."""
    assert len(SIGN_LABELS) == len(set(SIGN_LABELS)), (
        "Duplicate label names found in SIGN_LABELS. "
        "Each of the 43 classes must have a distinct name."
    )


def test_sign_labels_no_empty_strings():
    """Every class must have a non-empty name so predictions are human-readable."""
    for i, label in enumerate(SIGN_LABELS):
        assert label.strip() != "", f"Empty label at index {i}"


def test_sign_labels_known_classes():
    """Spot-check a few well-known GTSRB classes by their standard index.

    GTSRB class indices are fixed by the official dataset specification.
    If these spot-checks fail, the entire label list may be shifted or scrambled.
    """
    # Class 0: Speed limit 20 km/h — always the first class in GTSRB
    assert "20" in SIGN_LABELS[0] or "speed" in SIGN_LABELS[0].lower(), (
        f"Class 0 should be 'Speed limit 20 km/h', got: '{SIGN_LABELS[0]}'"
    )
    # Class 14: Stop sign — one of the most recognisable traffic signs
    assert "stop" in SIGN_LABELS[14].lower(), (
        f"Class 14 should be 'Stop', got: '{SIGN_LABELS[14]}'"
    )


def test_save_class_mapping_creates_csv(tmp_path):
    """save_class_mapping writes a CSV file with all 43 class IDs."""
    save_class_mapping(tmp_path)

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) >= 1, "No CSV file was created by save_class_mapping"

    with csv_files[0].open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Header + 43 data rows
    data_rows = [r for r in rows if r and not r[0].startswith("#")]
    assert len(data_rows) >= 43, (
        f"Expected at least 43 rows in class mapping CSV, got {len(data_rows)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
