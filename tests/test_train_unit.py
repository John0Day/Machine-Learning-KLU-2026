import unittest
from pathlib import Path
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train import run_epoch


class TinyClassifier(nn.Module):
    """Small deterministic model used to test train/eval epoch behavior quickly."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 2),
        )

    def forward(self, x):
        return self.net(x)


class TrainUnitTests(unittest.TestCase):
    def setUp(self):
        """Create a tiny synthetic binary dataset for fast, stable unit tests."""
        torch.manual_seed(7)
        x = torch.randn(24, 3, 4, 4)
        y = torch.randint(0, 2, (24,))
        self.loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
        self.model = TinyClassifier()
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cpu")

    def test_run_epoch_eval_returns_valid_metrics(self):
        """Eval mode should produce finite non-negative loss and bounded accuracy."""
        loss, acc = run_epoch(
            model=self.model,
            loader=self.loader,
            criterion=self.criterion,
            device=self.device,
            optimizer=None,
            max_batches=0,
        )
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_run_epoch_train_updates_weights(self):
        """Train mode should perform optimizer steps and change at least one parameter."""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        before = [p.detach().clone() for p in self.model.parameters()]

        run_epoch(
            model=self.model,
            loader=self.loader,
            criterion=self.criterion,
            device=self.device,
            optimizer=optimizer,
            max_batches=0,
        )
        after = [p.detach().clone() for p in self.model.parameters()]

        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)))


if __name__ == "__main__":
    unittest.main()
