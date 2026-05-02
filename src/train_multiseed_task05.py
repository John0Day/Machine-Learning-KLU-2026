"""Run multi-seed benchmarking for selected Task 05 models.

This script repeatedly trains the selected models with different random seeds
and stores aggregated metrics (mean/std) for robust model comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from model import BaselineCNN
from model_improved import DeepCNN, MobileNetTransfer, StrideCNN
from preprocessing import get_dataloaders
from train_improved import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed runner for Task 05 models")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/task05"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_specs(img_size: int):
    return [
        ("Baseline CNN", lambda: BaselineCNN(num_classes=43, input_size=img_size)),
        ("Deep CNN", lambda: DeepCNN(num_classes=43, input_size=img_size)),
        ("MobileNetV2", lambda: MobileNetTransfer(num_classes=43, freeze_backbone=False, input_size=img_size)),
        ("Stride CNN", lambda: StrideCNN(num_classes=43, input_size=img_size)),
    ]


def aggregate(records: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for row in records:
        by_model.setdefault(row["base_model"], []).append(row)

    summary: list[dict] = []
    for base_model, rows in by_model.items():
        test_accs = np.array([r["test_acc"] for r in rows], dtype=float)
        test_losses = np.array([r["test_loss"] for r in rows], dtype=float)
        val_accs = np.array([r["best_val_acc"] for r in rows], dtype=float)
        times = np.array([r["training_time_s"] for r in rows], dtype=float)
        params = int(rows[0]["num_params"])
        summary.append(
            {
                "model": base_model,
                "runs": len(rows),
                "test_acc_mean": float(test_accs.mean()),
                "test_acc_std": float(test_accs.std(ddof=0)),
                "test_loss_mean": float(test_losses.mean()),
                "test_loss_std": float(test_losses.std(ddof=0)),
                "best_val_acc_mean": float(val_accs.mean()),
                "best_val_acc_std": float(val_accs.std(ddof=0)),
                "training_time_mean_s": float(times.mean()),
                "training_time_std_s": float(times.std(ddof=0)),
                "num_params": params,
            }
        )

    # Sort best-first by mean test accuracy.
    summary.sort(key=lambda x: x["test_acc_mean"], reverse=True)
    return summary


def write_outputs(results_dir: Path, per_run: list[dict], summary: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = results_dir / "multiseed_per_run.json"
    with per_run_path.open("w", encoding="utf-8") as f:
        json.dump(per_run, f, indent=2)

    summary_path = results_dir / "multiseed_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = results_dir / "multiseed_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Task 05 Multi-Seed Comparison\n\n")
        f.write("| Model | Runs | Test Acc (mean ± std) | Test Loss (mean ± std) | Best Val Acc (mean ± std) | Params | Train Time (mean ± std) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            f.write(
                f"| {r['model']} | {r['runs']} | "
                f"{r['test_acc_mean']*100:.2f}% ± {r['test_acc_std']*100:.2f}% | "
                f"{r['test_loss_mean']:.4f} ± {r['test_loss_std']:.4f} | "
                f"{r['best_val_acc_mean']*100:.2f}% ± {r['best_val_acc_std']*100:.2f}% | "
                f"{r['num_params']:,} | "
                f"{r['training_time_mean_s']:.1f}s ± {r['training_time_std_s']:.1f}s |\n"
            )

    print(f"\nSaved per-run JSON   -> {per_run_path}")
    print(f"Saved summary JSON   -> {summary_path}")
    print(f"Saved summary table  -> {md_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Seeds: {args.seeds}")
    print("Loading data ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    per_run: list[dict] = []
    specs = model_specs(args.img_size)

    for seed in args.seeds:
        print(f"\n{'#'*72}\nSeed {seed}\n{'#'*72}")
        set_seed(seed)
        for base_name, factory in specs:
            model_name = f"{base_name} (seed {seed})"
            result = train_model(
                factory(),
                model_name,
                train_loader,
                val_loader,
                test_loader,
                device,
                args.epochs,
                args.lr,
                args.patience,
                args.models_dir,
            )
            compact = {
                "seed": seed,
                "base_model": base_name,
                "model_name": result["model_name"],
                "best_val_acc": result["best_val_acc"],
                "test_acc": result["test_acc"],
                "test_loss": result["test_loss"],
                "training_time_s": result["training_time_s"],
                "num_params": result["num_params"],
                "epochs_trained": result["epochs_trained"],
            }
            per_run.append(compact)

    summary = aggregate(per_run)
    write_outputs(args.results_dir, per_run, summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
