"""Comprehensive evaluation for Task 06.

Evaluates a selected model on the GTSRB test split and produces:
- test accuracy/loss
- confusion matrix (count + normalized)
- precision/recall/F1 per class
- bias analysis (frequent vs rare classes)
- misclassification example grid
- Grad-CAM examples
- robustness tests (gaussian noise + blur)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision.transforms import functional as TF

from model import BaselineCNN
from model_improved import DeepCNN, LeakyReLUCNN, MobileNetTransfer, StrideCNN
from preprocessing import get_dataloaders

_MEAN = torch.tensor((0.3337, 0.3064, 0.3171)).view(3, 1, 1)
_STD = torch.tensor((0.2672, 0.2564, 0.2629)).view(3, 1, 1)


@dataclass
class EvalOutputs:
    true_labels: torch.Tensor
    pred_labels: torch.Tensor
    confidences: torch.Tensor
    images: torch.Tensor
    test_loss: float
    test_acc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 06 evaluation pipeline")
    parser.add_argument("--model-type", type=str, default="deep", choices=["baseline", "deep", "mobilenet", "leaky", "stride"])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/task06"))
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-test-batches", type=int, default=0, help="0 means no limit")
    parser.add_argument("--num-misclassified", type=int, default=20)
    parser.add_argument("--num-gradcam", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--blur-kernel-size", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.img_size <= 0:
        raise ValueError("--img-size must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.max_test_batches < 0:
        raise ValueError("--max-test-batches must be >= 0")
    if args.num_misclassified <= 0:
        raise ValueError("--num-misclassified must be > 0")
    if args.num_gradcam <= 0:
        raise ValueError("--num-gradcam must be > 0")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be >= 0")
    if args.blur_kernel_size <= 0 or args.blur_kernel_size % 2 == 0:
        raise ValueError("--blur-kernel-size must be a positive odd integer (e.g. 3, 5, 7)")


def resolve_device(device_arg: str) -> torch.device:
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


def default_model_path(model_type: str) -> Path:
    mapping = {
        "baseline": Path("models/baseline_cnn.pth"),
        "deep": Path("models/deep_cnn.pth"),
        "mobilenet": Path("models/mobilenetv2.pth"),
        "leaky": Path("models/leakyrelu_cnn.pth"),
        "stride": Path("models/stride_cnn.pth"),
    }
    return mapping[model_type]


def build_model(model_type: str, img_size: int) -> nn.Module:
    if model_type == "baseline":
        return BaselineCNN(num_classes=43, input_size=img_size)
    if model_type == "deep":
        return DeepCNN(num_classes=43, input_size=img_size)
    if model_type == "mobilenet":
        return MobileNetTransfer(num_classes=43, freeze_backbone=False, input_size=img_size)
    if model_type == "leaky":
        return LeakyReLUCNN(num_classes=43, input_size=img_size)
    if model_type == "stride":
        return StrideCNN(num_classes=43, input_size=img_size)
    raise ValueError(f"Unsupported model type: {model_type}")


def load_weights(model: nn.Module, model_path: Path, device: torch.device) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images * std + mean).clamp(0, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(images.device)
    std = _STD.to(images.device)
    return (images - mean) / std


def load_class_names() -> List[str]:
    candidates = [
        Path("results/task03/class_mapping.csv"),
        Path("results/class_mapping.csv"),
    ]
    for path in candidates:
        if path.exists():
            rows = path.read_text(encoding="utf-8").strip().splitlines()
            names: List[str] = []
            for row in rows[1:]:
                parts = row.split(",", 1)
                if len(parts) == 2:
                    names.append(parts[1])
            if len(names) == 43:
                return names
    return [f"Class {i}" for i in range(43)]


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> EvalOutputs:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_true: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []
    all_conf: List[torch.Tensor] = []
    all_images: List[torch.Tensor] = []

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (pred == labels).sum().item()
            total_samples += batch_size

            all_true.append(labels.cpu())
            all_pred.append(pred.cpu())
            all_conf.append(conf.cpu())
            all_images.append(images.cpu())

            if max_batches > 0 and batch_idx >= max_batches:
                break

    true_labels = torch.cat(all_true, dim=0)
    pred_labels = torch.cat(all_pred, dim=0)
    confidences = torch.cat(all_conf, dim=0)
    images = torch.cat(all_images, dim=0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return EvalOutputs(true_labels, pred_labels, confidences, images, avg_loss, accuracy)


def save_confusion_matrices(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: Sequence[str],
    out_dir: Path,
) -> Dict[str, str]:
    cm = confusion_matrix(true_labels.numpy(), pred_labels.numpy(), labels=list(range(len(class_names))))

    count_path = out_dir / "confusion_matrix_counts.png"
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Task 06: Confusion Matrix (Counts)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(count_path, dpi=170)
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    norm_path = out_dir / "confusion_matrix_normalized.png"
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm_norm, cmap="magma", cbar=True, vmin=0.0, vmax=1.0)
    plt.title("Task 06: Confusion Matrix (Row-Normalized)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(norm_path, dpi=170)
    plt.close()

    return {"counts": str(count_path), "normalized": str(norm_path)}


def save_classification_report(
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: Sequence[str],
    out_dir: Path,
) -> Dict[str, str]:
    report_dict = classification_report(
        true_labels.numpy(),
        pred_labels.numpy(),
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        true_labels.numpy(),
        pred_labels.numpy(),
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    )

    json_path = out_dir / "classification_report.json"
    txt_path = out_dir / "classification_report.txt"
    csv_path = out_dir / "classification_report_per_class.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    txt_path.write_text(report_txt, encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1_score", "support"])
        for class_id, class_name in enumerate(class_names):
            row = report_dict.get(class_name, {})
            writer.writerow(
                [
                    class_id,
                    class_name,
                    float(row.get("precision", 0.0)),
                    float(row.get("recall", 0.0)),
                    float(row.get("f1-score", 0.0)),
                    int(row.get("support", 0)),
                ]
            )

    return {"json": str(json_path), "txt": str(txt_path), "csv": str(csv_path)}


def _extract_labels_from_dataset(dataset) -> List[int]:
    """Extract labels without iterating augmented DataLoader batches.

    Handles:
    - preprocessing._TransformedSubset (has `.subset`)
    - torch.utils.data.Subset (has `.dataset` + `.indices`)
    - preprocessing.GTSRBDataset (has `.samples`)
    """
    # Custom transformed subset wrapper used in preprocessing.py
    if hasattr(dataset, "subset"):
        return _extract_labels_from_dataset(dataset.subset)

    # torch.utils.data.Subset
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_dataset = dataset.dataset
        indices = [int(i) for i in dataset.indices]
        if hasattr(base_dataset, "samples"):
            return [int(base_dataset.samples[i][1]) for i in indices]
        base_labels = _extract_labels_from_dataset(base_dataset)
        return [int(base_labels[i]) for i in indices]

    # Base GTSRB dataset
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]

    # Fallback for unexpected dataset types
    labels: List[int] = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(int(label))
    return labels


def compute_class_counts_from_dataset(dataset, num_classes: int = 43) -> np.ndarray:
    labels = _extract_labels_from_dataset(dataset)
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes)
    return counts.astype(np.int64)


def save_bias_analysis(
    train_loader: torch.utils.data.DataLoader,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: Sequence[str],
    out_dir: Path,
) -> Dict[str, object]:
    train_counts = compute_class_counts_from_dataset(train_loader.dataset, num_classes=len(class_names))
    cm = confusion_matrix(true_labels.numpy(), pred_labels.numpy(), labels=list(range(43)))
    class_support = cm.sum(axis=1)
    class_acc = np.full(len(class_names), np.nan, dtype=np.float64)
    present_mask = class_support > 0
    diag = np.diag(cm)
    class_acc[present_mask] = diag[present_mask] / class_support[present_mask]

    ranked = np.argsort(train_counts)
    group_size = max(1, len(ranked) // 4)
    rare_ids = ranked[:group_size]
    frequent_ids = ranked[-group_size:]

    rare_present_ids = [int(i) for i in rare_ids if present_mask[i]]
    frequent_present_ids = [int(i) for i in frequent_ids if present_mask[i]]

    rare_acc = float(np.nanmean(class_acc[rare_present_ids])) if rare_present_ids else None
    frequent_acc = float(np.nanmean(class_acc[frequent_present_ids])) if frequent_present_ids else None
    accuracy_gap_abs = None if (rare_acc is None or frequent_acc is None) else float(abs(frequent_acc - rare_acc))

    def class_entry(class_id: int) -> Dict[str, object]:
        support = int(class_support[class_id])
        test_acc = None if support == 0 else float(class_acc[class_id])
        return {
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "train_count": int(train_counts[class_id]),
            "test_support": support,
            "test_acc": test_acc,
        }

    summary = {
        "frequent_classes": [class_entry(int(i)) for i in frequent_ids],
        "rare_classes": [class_entry(int(i)) for i in rare_ids],
        "frequent_classes_present_in_test": len(frequent_present_ids),
        "rare_classes_present_in_test": len(rare_present_ids),
        "frequent_mean_acc": frequent_acc,
        "rare_mean_acc": rare_acc,
        "accuracy_gap_abs": accuracy_gap_abs,
    }

    json_path = out_dir / "bias_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bar_path = out_dir / "bias_analysis_mean_accuracy.png"
    rare_plot = 0.0 if rare_acc is None else rare_acc * 100
    frequent_plot = 0.0 if frequent_acc is None else frequent_acc * 100
    rare_color = "#bbbbbb" if rare_acc is None else "#f4a261"
    frequent_color = "#bbbbbb" if frequent_acc is None else "#2a9d8f"

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Rare classes", "Frequent classes"], [rare_plot, frequent_plot], color=[rare_color, frequent_color])
    plt.ylabel("Mean class accuracy (%)")
    plt.title("Task 06: Bias Check (Rare vs Frequent Classes)")
    plt.ylim(0, 100)
    labels = ["n/a" if rare_acc is None else f"{rare_plot:.2f}%", "n/a" if frequent_acc is None else f"{frequent_plot:.2f}%"]
    for bar, label in zip(bars, labels):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()

    return {"summary_path": str(json_path), "plot_path": str(bar_path), **summary}


def save_misclassification_grid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    confidences: torch.Tensor,
    class_names: Sequence[str],
    out_path: Path,
    num_items: int,
) -> int:
    mistakes = torch.where(true_labels != pred_labels)[0].tolist()
    if not mistakes:
        return 0
    mistakes = sorted(mistakes, key=lambda i: float(confidences[i]), reverse=True)[:num_items]

    n = len(mistakes)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.2 * rows))
    axes_arr = np.array(axes).reshape(-1)
    imgs = denormalize(images)

    for i, ax in enumerate(axes_arr):
        if i >= n:
            ax.axis("off")
            continue
        idx = mistakes[i]
        t = int(true_labels[idx].item())
        p = int(pred_labels[idx].item())
        c = float(confidences[idx].item())
        ax.imshow(imgs[idx].permute(1, 2, 0).numpy())
        ax.set_title(f"T:{t} {class_names[t][:18]}\nP:{p} {class_names[p][:18]} ({c:.2f})", fontsize=8)
        ax.axis("off")

    fig.suptitle("Task 06: Top-Confidence Misclassifications", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return n


def apply_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    denorm = denormalize(images)
    noisy = (denorm + torch.randn_like(denorm) * std).clamp(0, 1)
    return normalize(noisy)


def apply_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    denorm = denormalize(images)
    blurred = TF.gaussian_blur(denorm, kernel_size=[kernel_size, kernel_size], sigma=[1.0, 1.0])
    return normalize(blurred)


def evaluate_with_transform(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    transform_fn,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = transform_fn(images.to(device))
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_samples += bs

            if max_batches > 0 and batch_idx >= max_batches:
                break

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": int(total_samples),
    }


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max().clamp(min=1e-8))
        return cam.detach().cpu()


def save_gradcam_examples(
    model: nn.Module,
    device: torch.device,
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    confidences: torch.Tensor,
    class_names: Sequence[str],
    out_path: Path,
    num_items: int,
) -> int:
    indices = torch.where(true_labels != pred_labels)[0].tolist()
    if not indices:
        indices = list(range(len(images)))
    indices = sorted(indices, key=lambda i: float(confidences[i]), reverse=True)
    indices = indices[: min(num_items, len(indices))]
    if not indices:
        return 0

    gradcam = GradCAM(model, find_last_conv_layer(model))
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes_arr = np.array(axes).reshape(-1)
    imgs_denorm = denormalize(images)

    for i, ax in enumerate(axes_arr):
        if i >= len(indices):
            ax.axis("off")
            continue
        idx = indices[i]
        img_norm = images[idx : idx + 1].to(device)
        pred_class = int(pred_labels[idx].item())
        cam = gradcam.generate(img_norm, pred_class).numpy()
        img = imgs_denorm[idx].permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        t = int(true_labels[idx].item())
        p = int(pred_labels[idx].item())
        c = float(confidences[idx].item())
        ax.set_title(f"T:{t} {class_names[t][:16]}\nP:{p} {class_names[p][:16]} ({c:.2f})", fontsize=8)
        ax.axis("off")

    fig.suptitle("Task 06: Grad-CAM (error-focused examples)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return len(indices)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    class_names = load_class_names()
    model_path = args.model_path if args.model_path is not None else default_model_path(args.model_type)

    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {model_path}")

    random.seed(args.split_seed)
    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)

    train_loader, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.split_seed,
    )

    model = build_model(args.model_type, args.img_size).to(device)
    load_weights(model, model_path, device)

    eval_out = evaluate_model(model, test_loader, device, args.max_test_batches)
    print(f"Test loss: {eval_out.test_loss:.4f}")
    print(f"Test acc : {eval_out.test_acc:.4f}")

    cm_paths = save_confusion_matrices(eval_out.true_labels, eval_out.pred_labels, class_names, args.results_dir)
    report_paths = save_classification_report(eval_out.true_labels, eval_out.pred_labels, class_names, args.results_dir)
    bias_summary = save_bias_analysis(train_loader, eval_out.true_labels, eval_out.pred_labels, class_names, args.results_dir)

    misclf_path = args.results_dir / "misclassifications_top_confidence.png"
    n_misclf_plotted = save_misclassification_grid(
        eval_out.images,
        eval_out.true_labels,
        eval_out.pred_labels,
        eval_out.confidences,
        class_names,
        misclf_path,
        args.num_misclassified,
    )

    gradcam_path = args.results_dir / "gradcam_examples.png"
    n_gradcam = save_gradcam_examples(
        model,
        device,
        eval_out.images,
        eval_out.true_labels,
        eval_out.pred_labels,
        eval_out.confidences,
        class_names,
        gradcam_path,
        args.num_gradcam,
    )

    noise_eval = evaluate_with_transform(
        model,
        test_loader,
        device,
        transform_fn=lambda x: apply_noise(x, args.noise_std),
        max_batches=args.max_test_batches,
    )
    blur_eval = evaluate_with_transform(
        model,
        test_loader,
        device,
        transform_fn=lambda x: apply_blur(x, args.blur_kernel_size),
        max_batches=args.max_test_batches,
    )

    robustness = {
        "clean": {"loss": eval_out.test_loss, "acc": eval_out.test_acc, "num_samples": int(len(eval_out.true_labels))},
        "gaussian_noise": noise_eval,
        "gaussian_blur": blur_eval,
        "noise_std": args.noise_std,
        "blur_kernel_size": args.blur_kernel_size,
    }
    robustness_path = args.results_dir / "robustness_metrics.json"
    with robustness_path.open("w", encoding="utf-8") as f:
        json.dump(robustness, f, indent=2)

    summary = {
        "model_type": args.model_type,
        "model_path": str(model_path),
        "device": str(device),
        "split_seed": args.split_seed,
        "num_test_samples": int(len(eval_out.true_labels)),
        "test_loss": eval_out.test_loss,
        "test_acc": eval_out.test_acc,
        "num_wrong": int((eval_out.true_labels != eval_out.pred_labels).sum().item()),
        "confusion_matrix_counts": cm_paths["counts"],
        "confusion_matrix_normalized": cm_paths["normalized"],
        "classification_report_json": report_paths["json"],
        "classification_report_txt": report_paths["txt"],
        "classification_report_csv": report_paths["csv"],
        "bias_analysis_json": bias_summary["summary_path"],
        "bias_plot": bias_summary["plot_path"],
        "misclassifications_plot": str(misclf_path),
        "misclassifications_plotted": n_misclf_plotted,
        "gradcam_plot": str(gradcam_path),
        "gradcam_items": n_gradcam,
        "robustness_json": str(robustness_path),
    }
    summary_path = args.results_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTask 06 outputs written to:")
    print(f"  - {args.results_dir}")
    print("\nKey metrics:")
    print(f"  clean accuracy      : {eval_out.test_acc:.4f}")
    print(f"  noisy accuracy      : {noise_eval['acc']:.4f}")
    print(f"  blurred accuracy    : {blur_eval['acc']:.4f}")
    print(f"  wrong classifications: {summary['num_wrong']}")
    print(f"  summary json        : {summary_path}")


if __name__ == "__main__":
    main()
