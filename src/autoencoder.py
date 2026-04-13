"""Autoencoder for anomaly detection on GTSRB (Task 05).

Implements the Autoencoder concept from Lecture 7:
- Encoder compresses images to a low-dimensional latent vector
- Decoder reconstructs the image from the latent vector
- Reconstruction error (MSE) is used to detect anomalies:
  Known signs → low reconstruction error
  Unknown/damaged signs → high reconstruction error

Usage:
    # Train
    python src/autoencoder.py --mode train --device mps

    # Evaluate anomaly detection
    python src/autoencoder.py --mode evaluate --device mps
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
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from preprocessing import get_dataloaders


# ---------------------------------------------------------------------------
# Autoencoder Architecture (Lecture 7)
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """Convolutional encoder: image → latent vector.

    Compresses 3×32×32 = 3072 values down to latent_dim values.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3×32×32 → 32×32×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 32×16×16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×16×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 64×8×8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → 128×8×8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → 128×4×4
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    """Convolutional decoder: latent vector → reconstructed image.

    Mirrors the encoder structure using transposed convolutions.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 4×4 → 8×8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 8×8 → 16×16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),     # 16×16 → 32×32
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    """Full Autoencoder = Encoder + Decoder.

    Loss function: Mean Squared Error between input and reconstruction.
    Low MSE = image is known/normal.
    High MSE = image is unknown/anomalous.
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=[1, 2, 3])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: ConvAutoencoder,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    models_dir: Path,
    results_dir: Path,
) -> None:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_path = models_dir / "autoencoder.pth"

    print(f"\nTraining Autoencoder for up to {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for images, _ in train_loader:   # labels not used!
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon = model(images)
                total_loss += criterion(recon, images).item() * images.size(0)
        val_loss = total_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  → Best model saved (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    # Plot loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = results_dir / "autoencoder_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved → {path}")
    print(f"Best model saved → {best_path}")


# ---------------------------------------------------------------------------
# Anomaly Detection Evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly_detection(
    model: ConvAutoencoder,
    val_loader,
    device: torch.device,
    results_dir: Path,
    threshold_percentile: float = 95.0,
) -> None:
    """Visualise reconstruction errors and anomaly threshold.

    Images with reconstruction error above the threshold are flagged
    as anomalies (unknown or damaged signs).
    """
    model.eval()
    errors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            err = model.reconstruction_error(images)
            errors.extend(err.cpu().numpy())
            labels_list.extend(labels.numpy())

    errors = np.array(errors)
    labels_arr = np.array(labels_list)
    threshold = np.percentile(errors, threshold_percentile)

    print(f"\nReconstruction Error Stats:")
    print(f"  Mean   : {errors.mean():.6f}")
    print(f"  Std    : {errors.std():.6f}")
    print(f"  Min    : {errors.min():.6f}")
    print(f"  Max    : {errors.max():.6f}")
    print(f"  Threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    print(f"  Flagged as anomaly: {(errors > threshold).sum()} / {len(errors)} samples")

    # Plot error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=80, edgecolor="none", alpha=0.7, label="Reconstruction Error")
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"Anomaly Threshold ({threshold_percentile}th percentile)")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = results_dir / "autoencoder_error_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Error distribution saved → {path}")

    # Show best and worst reconstructions
    _plot_reconstructions(model, val_loader, device, errors, results_dir)


def _plot_reconstructions(
    model: ConvAutoencoder,
    loader,
    device: torch.device,
    errors: np.ndarray,
    results_dir: Path,
    n: int = 8,
) -> None:
    """Save a grid of best and worst reconstructions."""
    all_images = []
    for images, _ in loader:
        all_images.append(images)
        if sum(len(x) for x in all_images) >= len(errors):
            break
    all_images = torch.cat(all_images, dim=0)[:len(errors)]

    best_idx = np.argsort(errors)[:n]
    worst_idx = np.argsort(errors)[-n:]

    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 10))
    fig.suptitle("Autoencoder Reconstructions\nTop: Best (normal) | Bottom: Worst (anomalous)", fontsize=12)

    mean = torch.tensor([0.3337, 0.3064, 0.3171]).view(3, 1, 1)
    std  = torch.tensor([0.2672, 0.2564, 0.2629]).view(3, 1, 1)

    def denorm(t):
        return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    with torch.no_grad():
        for i, idx in enumerate(best_idx):
            img = all_images[idx:idx+1].to(device)
            recon = model(img).cpu().squeeze(0)
            orig = all_images[idx]
            axes[0, i].imshow(denorm(orig))
            axes[0, i].set_title(f"err={errors[idx]:.4f}", fontsize=7)
            axes[0, i].axis("off")
            axes[1, i].imshow(denorm(recon))
            axes[1, i].axis("off")

        for i, idx in enumerate(worst_idx):
            img = all_images[idx:idx+1].to(device)
            recon = model(img).cpu().squeeze(0)
            orig = all_images[idx]
            axes[2, i].imshow(denorm(orig))
            axes[2, i].set_title(f"err={errors[idx]:.4f}", fontsize=7)
            axes[2, i].axis("off")
            axes[3, i].imshow(denorm(recon))
            axes[3, i].axis("off")

    axes[0, 0].set_ylabel("Original\n(best)", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed\n(best)", fontsize=9)
    axes[2, 0].set_ylabel("Original\n(worst)", fontsize=9)
    axes[3, 0].set_ylabel("Reconstructed\n(worst)", fontsize=9)

    fig.tight_layout()
    path = results_dir / "autoencoder_reconstructions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Reconstruction grid saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoencoder anomaly detection")
    parser.add_argument("--mode",        choices=["train", "evaluate", "both"], default="both")
    parser.add_argument("--data-root",   type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir",  type=Path, default=Path("models"))
    parser.add_argument("--model-path",  type=Path, default=Path("models/autoencoder.pth"))
    parser.add_argument("--img-size",    type=int,  default=32)
    parser.add_argument("--batch-size",  type=int,  default=64)
    parser.add_argument("--epochs",      type=int,  default=30)
    parser.add_argument("--lr",          type=float,default=0.001)
    parser.add_argument("--patience",    type=int,  default=5)
    parser.add_argument("--latent-dim",  type=int,  default=128)
    parser.add_argument("--device",      type=str,  default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    print("Loading data …")
    train_loader, val_loader, _ = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = ConvAutoencoder(latent_dim=args.latent_dim)

    if args.mode in ("train", "both"):
        train_autoencoder(
            model, train_loader, val_loader, device,
            args.epochs, args.lr, args.patience,
            args.models_dir, args.results_dir,
        )

    if args.mode in ("evaluate", "both"):
        print(f"\nLoading autoencoder from {args.model_path} …")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        evaluate_anomaly_detection(model, val_loader, device, args.results_dir)

    print("\n✓ Autoencoder complete!")
