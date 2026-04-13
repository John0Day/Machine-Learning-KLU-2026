"""Latent space visualisation using t-SNE (Task 05).

Extracts feature vectors from the trained CNN (before the final classifier
layer) and projects them to 2D using t-SNE. Shows which traffic sign classes
cluster together in the learned feature space.

Lecture 7 introduces t-SNE/UMAP for dimensionality reduction and latent
space visualisation in the context of Autoencoders.

Usage:
    python src/visualize.py
    python src/visualize.py --model-path models/baseline_seed-42.pth --device mps
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from model import BaselineCNN
from preprocessing import get_dataloaders


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors (before final Linear layer) and labels.

    Args:
        model:       Trained CNN model.
        loader:      DataLoader to extract features from.
        device:      Compute device.
        max_samples: Limit samples for faster t-SNE computation.

    Returns:
        Tuple of (features array [N, D], labels array [N]).
    """
    model.eval()

    # Hook the output of the second-to-last layer (before final Linear)
    features_list = []
    labels_list = []

    # Register forward hook on the classifier (up to Dropout, before last Linear)
    activation = {}

    def hook_fn(module, input, output):
        activation["features"] = output.detach().cpu()

    # Hook after the first Linear + ReLU + Dropout in classifier
    handle = model.classifier[2].register_forward_hook(hook_fn)  # ReLU output

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            model(images)  # triggers hook
            features_list.append(activation["features"])
            labels_list.append(labels)

            if sum(len(f) for f in features_list) >= max_samples:
                break

    handle.remove()

    features = torch.cat(features_list, dim=0)[:max_samples].numpy()
    labels = torch.cat(labels_list, dim=0)[:max_samples].numpy()
    return features, labels


# ---------------------------------------------------------------------------
# t-SNE + plotting
# ---------------------------------------------------------------------------

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    results_dir: Path,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> None:
    """Run t-SNE and save the 2D scatter plot.

    Args:
        features:    Feature matrix [N, D].
        labels:      Class labels [N].
        results_dir: Directory to save the plot.
        perplexity:  t-SNE perplexity parameter.
        n_iter:      Number of t-SNE iterations.
    """
    print(f"Running t-SNE on {len(features)} samples (perplexity={perplexity}) …")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embedded = tsne.fit_transform(features)

    # Color map with 43 distinct colors
    cmap = cm.get_cmap("tab20", 43)
    colors = [cmap(i) for i in range(43)]

    fig, ax = plt.subplots(figsize=(14, 10))
    for class_id in range(43):
        mask = labels == class_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=8,
            color=colors[class_id],
            alpha=0.7,
            label=str(class_id),
        )

    ax.set_title("t-SNE Visualisation of CNN Feature Space (Validation Set)", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(
        title="Class ID",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=6,
        ncol=2,
    )
    fig.tight_layout()

    path = results_dir / "tsne_feature_space.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE latent space visualisation")
    parser.add_argument("--model-path",  type=Path, default=Path("models/baseline_seed-42.pth"))
    parser.add_argument("--data-root",   type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--img-size",    type=int,  default=32)
    parser.add_argument("--batch-size",  type=int,  default=64)
    parser.add_argument("--max-samples", type=int,  default=2000)
    parser.add_argument("--perplexity",  type=int,  default=30)
    parser.add_argument("--device",      type=str,  default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
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

    # Load model
    print(f"Loading model from {args.model_path} …")
    model = BaselineCNN(num_classes=43, input_size=args.img_size).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print("Model loaded.")

    # Data
    _, val_loader, _ = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # Extract features & plot
    features, labels = extract_features(model, val_loader, device, args.max_samples)
    print(f"Extracted features: {features.shape}")
    plot_tsne(features, labels, args.results_dir, perplexity=args.perplexity)
    print("✓ Visualisation complete")
