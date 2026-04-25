"""Hyperparameter tuning with Optuna for GTSRB traffic sign classification.

Uses Bayesian optimisation (TPE sampler) to search over learning rate,
dropout, batch size, and optimiser type.  Each trial trains for a fixed
number of epochs and returns the best validation accuracy found.

The best hyperparameters are printed at the end and saved to
results/best_hyperparams.json.

Usage
-----
    python src/tune.py                          # 30 trials, 10 epochs each
    python src/tune.py --n-trials 50 --epochs 15
    python src/tune.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Matplotlib backend must be set before any import of matplotlib.pyplot
# ---------------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)

from preprocessing import get_dataloaders
from model_improved import StrideCNN  # Used for tuning (Task 05); DeepCNN achieved highest overall accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer=None,
    device: torch.device = torch.device("cpu"),
    training: bool = True,
) -> tuple[float, float]:
    """Run one epoch.  Returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(data_root: Path, n_epochs: int, device: torch.device):
    """Return an Optuna objective that trains StrideCNN with sampled params."""

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ──────────────────────────────────────────
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # ── Dataloaders ─────────────────────────────────────────────────────
        train_loader, val_loader, _ = get_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=0,
        )

        # ── Model ───────────────────────────────────────────────────────────
        model = StrideCNN(num_classes=43, dropout=dropout).to(device)
        criterion = nn.CrossEntropyLoss()

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        best_val_acc = 0.0

        for epoch in range(n_epochs):
            run_epoch(model, train_loader, criterion, optimizer, device, training=True)
            _, val_acc = run_epoch(model, val_loader, criterion, None, device, training=False)

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Pruning: stop unpromising trials early
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_acc

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per trial (default: 10)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hyperparameter Tuning — Optuna (Bayesian Optimisation)")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Epochs/trial: {args.epochs}")
    print(f"  Data root : {args.data_root}")
    print()

    # ── Create study ────────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)          # Bayesian (Tree Parzen)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3) # Kill bad trials early

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="gtsrb_stride_cnn",
    )

    objective = make_objective(args.data_root, args.epochs, device)

    # Suppress per-trial log spam — show only summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Running {args.n_trials} trials …\n")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ─────────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Validation Accuracy : {best.value * 100:.2f} %")
    print(f"  Trial number        : {best.number}")
    print("  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<20} = {v}")

    # Save JSON
    result = {
        "best_val_accuracy": best.value,
        "trial_number": best.number,
        "hyperparameters": best.params,
    }
    out_json = args.output_dir / "best_hyperparams.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_json}")

    # ── Plots ────────────────────────────────────────────────────────────────
    out_plot = args.output_dir / "tuning_results.png"
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # History plot
        ax1 = plot_optimization_history(study)
        axes[0].set_title("Optimization History")
        # Copy lines from optuna figure into our subplot
        for line in ax1.get_lines():
            axes[0].plot(line.get_xdata(), line.get_ydata(),
                         color=line.get_color(), label=line.get_label())
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].legend()

        # Importance plot — build manually from importances dict
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())
        values = list(importances.values())
        axes[1].barh(names, values)
        axes[1].set_title("Hyperparameter Importance")
        axes[1].set_xlabel("Importance")

        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        plt.close("all")
        print(f"  Plot     → {out_plot}")
    except Exception as e:
        print(f"  Plot skipped ({e})")
    print()

    # ── Top-5 trials table ───────────────────────────────────────────────────
    print("Top 5 Trials:")
    print(f"  {'Rank':<6} {'Val Acc':>8}  {'lr':>10}  {'dropout':>8}  "
          f"{'batch':>6}  {'optimizer':<8}  {'wd':>10}")
    print("  " + "-" * 70)

    completed = [t for t in study.trials if t.value is not None]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    for rank, t in enumerate(top5, 1):
        p = t.params
        print(
            f"  {rank:<6} {t.value * 100:>7.2f}%  "
            f"{p.get('lr', 0):>10.5f}  "
            f"{p.get('dropout', 0):>8.3f}  "
            f"{p.get('batch_size', 0):>6}  "
            f"{p.get('optimizer', '?'):<8}  "
            f"{p.get('weight_decay', 0):>10.6f}"
        )

    print()
    print("Done! Use the best hyperparameters above to retrain your final model.")


if __name__ == "__main__":
    main()
