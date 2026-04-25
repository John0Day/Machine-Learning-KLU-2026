import tempfile
import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demo_offline import load_checkpoint


class DemoOfflineUnitTests(unittest.TestCase):
    def test_load_checkpoint_with_training_checkpoint_format(self):
        """Support checkpoints saved with metadata + model_state_dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            payload = {"model_state_dict": {"weight": torch.tensor([1.0])}, "img_size": 64}
            torch.save(payload, path)

            checkpoint, img_size = load_checkpoint(path, torch.device("cpu"))
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(img_size, 64)

    def test_load_checkpoint_with_plain_state_dict_format(self):
        """Support older/plain state_dict files by normalizing to expected structure."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.pth"
            payload = {"weight": torch.tensor([1.0])}
            torch.save(payload, path)

            checkpoint, img_size = load_checkpoint(path, torch.device("cpu"))
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(img_size, 32)

    def test_load_checkpoint_missing_file_raises(self):
        """Missing checkpoint should fail loudly with FileNotFoundError."""
        missing = Path("/tmp/this_file_does_not_exist_for_test_checkpoint.pth")
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(missing, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
