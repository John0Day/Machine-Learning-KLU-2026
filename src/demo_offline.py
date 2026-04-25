"""Offline visual demo for the GTSRB baseline classifier.

Generates presentation-ready artifacts without webcam input:
- test prediction grid
- high-confidence misclassification grid
- confusion matrix heatmap
- summary JSON
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import List, Sequence

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from model import BaselineCNN
from preprocessing import get_dataloaders

_MEAN = torch.tensor((0.3337, 0.3064, 0.3171)).view(3, 1, 1)
_STD = torch.tensor((0.2672, 0.2564, 0.2629)).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline demo for baseline GTSRB model")
    parser.add_argument("--model", type=Path, default=Path("models/baseline.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo_offline"))
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-mistakes", type=int, default=20)
    parser.add_argument("--max-test-batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve requested runtime backend with explicit fallbacks."""
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


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[dict, int]:
    """Load checkpoint and normalize legacy/new formats to a common dict."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, int(checkpoint.get("img_size", 32))

    if isinstance(checkpoint, dict):
        return {"model_state_dict": checkpoint, "img_size": 32}, 32

    raise RuntimeError("Unsupported checkpoint format")


def load_class_names() -> list[str]:
    csv_path = Path("results/class_mapping.csv")
    if csv_path.exists():
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()[1:]
        names = []
        for line in lines:
            parts = line.split(",", 1)
            if len(parts) == 2:
                names.append(parts[1])
        if len(names) == 43:
            return names
    return [f"Class {i}" for i in range(43)]


def denormalize(img: torch.Tensor) -> torch.Tensor:
    return (img * _STD + _MEAN).clamp(0, 1)


def shorten(label: str, max_len: int = 24) -> str:
    return label if len(label) <= max_len else (label[: max_len - 1] + "…")


def collect_test_outputs(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect predictions, confidences, and images for downstream plotting."""
    model.eval()

    all_images: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    all_pred: list[torch.Tensor] = []
    all_conf: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, start=1):
            # Keep tensors in normalized form; plotting code denormalizes on demand.
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            all_images.append(images.cpu())
            all_true.append(labels.cpu())
            all_pred.append(pred.cpu())
            all_conf.append(conf.cpu())

            if max_batches > 0 and batch_idx >= max_batches:
                break

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_true, dim=0),
        torch.cat(all_pred, dim=0),
        torch.cat(all_conf, dim=0),
    )


def plot_prediction_grid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    confidences: torch.Tensor,
    indices: Sequence[int],
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    if not indices:
        return

    n = len(indices)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue

        idx = indices[i]
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        t = int(true_labels[idx])
        p = int(pred_labels[idx])
        c = float(confidences[idx])

        ax.imshow(img)
        ax.set_title(
            f"T:{t} {shorten(class_names[t], 18)}\nP:{p} {shorten(class_names[p], 18)} ({c:.2f})",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(true_labels.numpy(), pred_labels.numpy(), labels=list(range(len(class_names))))

    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Baseline CNN Confusion Matrix (Test Split)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    class_names = load_class_names()

    checkpoint, ckpt_img_size = load_checkpoint(args.model, device)
    # Use image size embedded in checkpoint when available to avoid shape mismatch.
    img_size = ckpt_img_size if ckpt_img_size else args.img_size

    model = BaselineCNN(num_classes=43, input_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.split_seed,
    )

    images, true_labels, pred_labels, confidences = collect_test_outputs(
        model=model,
        test_loader=test_loader,
        device=device,
        max_batches=args.max_test_batches,
    )

    accuracy = float((pred_labels == true_labels).float().mean().item()) if len(true_labels) else 0.0

    rng = random.Random(args.sample_seed)
    # Random subset: lightweight qualitative overview of normal predictions.
    sample_indices = list(range(len(true_labels)))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[: min(args.num_samples, len(sample_indices))]

    mistakes = [
        i for i in range(len(true_labels))
        if int(pred_labels[i]) != int(true_labels[i])
    ]
    # Focus error grid on high-confidence mistakes: most informative failure cases.
    mistakes_sorted = sorted(mistakes, key=lambda i: float(confidences[i]), reverse=True)
    mistakes_indices = mistakes_sorted[: min(args.num_mistakes, len(mistakes_sorted))]

    grid_path = args.output_dir / "predictions_grid.png"
    mistakes_path = args.output_dir / "misclassifications_top_confidence.png"
    cm_path = args.output_dir / "confusion_matrix.png"
    summary_path = args.output_dir / "summary.json"

    plot_prediction_grid(
        images=images,
        true_labels=true_labels,
        pred_labels=pred_labels,
        confidences=confidences,
        indices=sample_indices,
        class_names=class_names,
        out_path=grid_path,
        title="Baseline CNN: Random Test Predictions",
    )

    if mistakes_indices:
        plot_prediction_grid(
            images=images,
            true_labels=true_labels,
            pred_labels=pred_labels,
            confidences=confidences,
            indices=mistakes_indices,
            class_names=class_names,
            out_path=mistakes_path,
            title="Baseline CNN: High-Confidence Misclassifications",
        )

    plot_confusion_matrix(true_labels, pred_labels, class_names, cm_path)

    summary = {
        "model": str(args.model),
        "device": str(device),
        "img_size": img_size,
        "num_test_samples_used": int(len(true_labels)),
        "accuracy": accuracy,
        "num_errors": int((pred_labels != true_labels).sum().item()),
        "prediction_grid": str(grid_path),
        "misclassifications_grid": str(mistakes_path if mistakes_indices else ""),
        "confusion_matrix": str(cm_path),
        "split_seed": args.split_seed,
        "sample_seed": args.sample_seed,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Offline demo complete.")
    print(f"Test accuracy                 : {accuracy:.4f}")
    print(f"Prediction grid               : {grid_path}")
    if mistakes_indices:
        print(f"Misclassification grid        : {mistakes_path}")
    else:
        print("Misclassification grid        : no errors in evaluated subset")
    print(f"Confusion matrix              : {cm_path}")
    print(f"Summary JSON                  : {summary_path}")


if __name__ == "__main__":
    main()
