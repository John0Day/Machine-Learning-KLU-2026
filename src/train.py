"""Train baseline CNN on local GTSRB data (Task 04)."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

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
from preprocessing import get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline CNN on GTSRB")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run label for output filenames (default: seed-<seed>)",
    )
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve compute backend from CLI selection."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Run one full pass over a loader in train or eval mode."""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                # `set_to_none=True` saves memory and can improve performance.
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                # Backprop + optimizer step happen only during training pass.
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            if max_batches > 0 and batch_idx >= max_batches:
                break

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    out_path: Path,
) -> None:
    """Save loss and accuracy curves side by side."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train Accuracy")
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    # Single seed controls both split reproducibility and model initialization.
    torch.manual_seed(args.seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = BaselineCNN(num_classes=43, input_size=args.img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Plateau scheduler lowers LR when validation loss stalls.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    run_name = args.run_name.strip() if args.run_name.strip() else f"seed-{args.seed}"
    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_model_path = args.models_dir / f"baseline_{run_name}.pth"

    print(f"Training for up to {args.epochs} epochs (early stopping patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            # Save full training state so the best epoch can be restored exactly.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "img_size": args.img_size,
                },
                best_model_path,
            )
            print(f"  → New best model saved (val_acc={best_val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                # Stop once validation performance has not improved for `patience` epochs.
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    if best_model_path.exists():
        # Evaluate using the best validation checkpoint, not the last epoch state.
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Keep the canonical path required by Task 04 while preserving
        # run-specific checkpoints for reproducibility.
        canonical_model_path = args.models_dir / "baseline.pth"
        if canonical_model_path != best_model_path:
            torch.save(checkpoint, canonical_model_path)

    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        max_batches=args.max_val_batches,
    )

    loss_plot_path = args.results_dir / f"baseline_curves_{run_name}.png"
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies, loss_plot_path)
    print(f"Training curves saved → {loss_plot_path}")

    history: Dict[str, object] = {
        "epochs": args.epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model_path": str(best_model_path),
        "canonical_model_path": str(args.models_dir / "baseline.pth"),
        "loss_plot_path": str(loss_plot_path),
    }

    history["seed"] = args.seed
    history["run_name"] = run_name

    history_path = args.results_dir / f"baseline_history_{run_name}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Saved best model    -> {best_model_path}")
    print(f"Saved loss curve    -> {loss_plot_path}")
    print(f"Saved history       -> {history_path}")
    print(f"Final test metrics  -> loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
