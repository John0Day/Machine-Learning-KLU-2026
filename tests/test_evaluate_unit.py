import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Subset

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import compute_class_counts_from_dataset, save_bias_analysis, validate_args


class FakeBaseDataset:
    """Minimal dataset with `.samples` compatible with evaluate helpers."""

    def __init__(self, labels):
        self.samples = [(Path(f"img_{i}.ppm"), int(label)) for i, label in enumerate(labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Content is irrelevant for these tests; only labels matter.
        return torch.zeros(3, 32, 32), self.samples[idx][1]


class FakeTransformedSubset:
    """Mimics preprocessing._TransformedSubset by exposing `.subset`."""

    def __init__(self, subset):
        self.subset = subset


class EvaluateUnitTests(unittest.TestCase):
    def test_validate_args_rejects_even_blur_kernel(self):
        """Blur kernel must be odd for gaussian blur to be valid."""
        args = SimpleNamespace(
            img_size=32,
            batch_size=64,
            num_workers=0,
            max_test_batches=0,
            num_misclassified=10,
            num_gradcam=5,
            noise_std=0.1,
            blur_kernel_size=4,
        )
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_validate_args_accepts_valid_values(self):
        """Valid CLI args should pass without raising validation errors."""
        args = SimpleNamespace(
            img_size=32,
            batch_size=64,
            num_workers=0,
            max_test_batches=0,
            num_misclassified=10,
            num_gradcam=5,
            noise_std=0.0,
            blur_kernel_size=5,
        )
        validate_args(args)  # should not raise

    def test_compute_class_counts_handles_subset_wrapper(self):
        """Label counting must work through nested subset wrappers."""
        base = FakeBaseDataset(labels=[0, 1, 1, 2, 2, 2])
        subset = Subset(base, [0, 2, 3, 5])  # labels -> [0,1,2,2]
        wrapped = FakeTransformedSubset(subset)
        counts = compute_class_counts_from_dataset(wrapped, num_classes=4)
        self.assertEqual(counts.tolist(), [1, 1, 2, 0])

    def test_save_bias_analysis_handles_missing_test_support(self):
        """Classes absent in test split should be exported with null accuracy."""
        # Create train labels with all 43 classes represented at least once.
        train_labels = [i for i in range(43)] + [42] * 10 + [41] * 8 + [0] * 2
        train_dataset = FakeBaseDataset(labels=train_labels)
        train_loader = SimpleNamespace(dataset=train_dataset)

        # Only class 0 appears in this small test sample -> many classes have no support.
        true_labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        pred_labels = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        class_names = [f"Class {i}" for i in range(43)]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = save_bias_analysis(train_loader, true_labels, pred_labels, class_names, out_dir)
            self.assertIn("summary_path", result)
            summary_path = Path(result["summary_path"])
            self.assertTrue(summary_path.exists())

            summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("frequent_classes_present_in_test", summary_json)
            self.assertIn("rare_classes_present_in_test", summary_json)
            # Ensure unsupported classes are not forced to 0.0 and remain explicit nulls.
            unsupported_entries = [
                row for row in (summary_json["frequent_classes"] + summary_json["rare_classes"])
                if row["test_support"] == 0
            ]
            self.assertTrue(len(unsupported_entries) > 0)
            self.assertTrue(any(row["test_acc"] is None for row in unsupported_entries))


if __name__ == "__main__":
    unittest.main()
