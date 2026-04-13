"""Train and compare improved models against the baseline (Task 05).

Trains DeepCNN and MobileNetV2 and saves a comparison table to results/.

Usage:
    python src/train_improved.py
    python src/train_improved.py --epochs 20 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from model import BaselineCNN
from model_improved import DeepCNN, MobileNetTransfer, LeakyReLUCNN, StrideCNN
from preprocessing import get_dataloaders


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One forward pass over the loader. Trains if optimizer is given."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    models_dir: Path,
) -> dict:
    """Train a model and return results dict."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_path = models_dir / f"{model_name.lower().replace(' ', '_')}.pth"
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  → New best saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    training_time = time.time() - start_time

    # Final test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTest Accuracy : {test_acc*100:.2f}%")
    print(f"Training Time : {training_time:.1f}s")
    print(f"Parameters    : {num_params:,}")

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "training_time_s": round(training_time, 1),
        "num_params": num_params,
        "epochs_trained": len(train_accs),
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ---------------------------------------------------------------------------
# Plotting & comparison table
# ---------------------------------------------------------------------------

def plot_comparison(results: list[dict], results_dir: Path) -> None:
    """Plot accuracy curves for all models side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        epochs = range(1, len(r["train_accs"]) + 1)
        ax.plot(epochs, [a * 100 for a in r["train_accs"]], label="Train")
        ax.plot(epochs, [a * 100 for a in r["val_accs"]], label="Val")
        ax.set_title(r["model_name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Model Comparison – Accuracy Curves", fontsize=14)
    fig.tight_layout()
    path = results_dir / "model_comparison_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nComparison curves saved → {path}")


def save_comparison_table(results: list[dict], results_dir: Path) -> None:
    """Save a markdown comparison table and JSON results."""
    # Markdown table
    md_path = results_dir / "model_comparison.md"
    with md_path.open("w") as f:
        f.write("# Model Comparison (Task 05)\n\n")
        f.write("| Model | Test Accuracy | Val Accuracy | Params | Training Time |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['model_name']} "
                f"| {r['test_acc']*100:.2f}% "
                f"| {r['best_val_acc']*100:.2f}% "
                f"| {r['num_params']:,} "
                f"| {r['training_time_s']:.0f}s |\n"
            )
    print(f"Comparison table saved → {md_path}")

    # JSON
    json_path = results_dir / "model_comparison.json"
    summary = [
        {k: v for k, v in r.items() if k not in ("train_accs", "val_accs", "train_losses", "val_losses")}
        for r in results
    ]
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison JSON saved  → {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved models for Task 05")
    parser.add_argument("--data-root",   type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir",  type=Path, default=Path("models"))
    parser.add_argument("--img-size",    type=int,  default=32)
    parser.add_argument("--batch-size",  type=int,  default=64)
    parser.add_argument("--epochs",      type=int,  default=20)
    parser.add_argument("--lr",          type=float,default=0.001)
    parser.add_argument("--patience",    type=int,  default=5)
    parser.add_argument("--device",      type=str,  default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-baseline",   action="store_true")
    parser.add_argument("--skip-deep",       action="store_true")
    parser.add_argument("--skip-mobilenet",  action="store_true")
    parser.add_argument("--skip-leaky",      action="store_true")
    parser.add_argument("--skip-stride",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading data …")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    results = []

    # Baseline (for reference)
    if not args.skip_baseline:
        r = train_model(
            BaselineCNN(num_classes=43, input_size=args.img_size),
            "Baseline CNN",
            train_loader, val_loader, test_loader,
            device, args.epochs, args.lr, args.patience, args.models_dir,
        )
        results.append(r)

    # Variant A – DeepCNN
    if not args.skip_deep:
        r = train_model(
            DeepCNN(num_classes=43, input_size=args.img_size),
            "Deep CNN",
            train_loader, val_loader, test_loader,
            device, args.epochs, args.lr, args.patience, args.models_dir,
        )
        results.append(r)

    # Variant B – MobileNetV2
    if not args.skip_mobilenet:
        r = train_model(
            MobileNetTransfer(num_classes=43, freeze_backbone=False, input_size=args.img_size),
            "MobileNetV2",
            train_loader, val_loader, test_loader,
            device, args.epochs, args.lr, args.patience, args.models_dir,
        )
        results.append(r)

    # Variant C – LeakyReLU CNN
    if not args.skip_leaky:
        r = train_model(
            LeakyReLUCNN(num_classes=43, input_size=args.img_size),
            "LeakyReLU CNN",
            train_loader, val_loader, test_loader,
            device, args.epochs, args.lr, args.patience, args.models_dir,
        )
        results.append(r)

    # Variant D – StrideCNN
    if not args.skip_stride:
        r = train_model(
            StrideCNN(num_classes=43, input_size=args.img_size),
            "Stride CNN",
            train_loader, val_loader, test_loader,
            device, args.epochs, args.lr, args.patience, args.models_dir,
        )
        results.append(r)

    # Save comparison
    plot_comparison(results, args.results_dir)
    save_comparison_table(results, args.results_dir)

    print("\n✓ Task 05 complete!")
