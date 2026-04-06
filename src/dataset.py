"""GTSRB dataset inspection and visualization utilities.

This script reads the official GTSRB training image annotations from a local
folder (downloaded from sid.erda.dk), computes basic dataset statistics, and
saves Task 02 artifacts to the results directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-klu-gtsrb"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())

import matplotlib

# Use a non-interactive backend so the script works in headless terminals.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Official GTSRB class names (43 classes).
SIGN_LABELS: List[str] = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5t",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5t",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local GTSRB training data")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing GTSRB_Final_Training_Images",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where plots and statistics will be written",
    )
    return parser.parse_args()


def resolve_images_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "GTSRB_Final_Training_Images" / "Final_Training" / "Images",
        data_root / "Final_Training" / "Images",
        data_root,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Could not find GTSRB image directory. Expected something like "
        "data/raw/GTSRB_Final_Training_Images/Final_Training/Images"
    )


def load_annotations(images_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    class_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {images_dir}")

    for class_dir in class_dirs:
        class_id = int(class_dir.name)
        gt_file = class_dir / f"GT-{class_dir.name}.csv"
        if not gt_file.exists():
            raise FileNotFoundError(f"Missing annotation file: {gt_file}")

        with gt_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                filename = row["Filename"]
                width = int(row["Width"])
                height = int(row["Height"])
                image_path = class_dir / filename

                records.append(
                    {
                        "class_id": class_id,
                        "class_name": SIGN_LABELS[class_id] if class_id < len(SIGN_LABELS) else f"Class {class_id}",
                        "width": width,
                        "height": height,
                        "image_path": str(image_path),
                    }
                )

    return records


def save_class_mapping(results_dir: Path) -> Path:
    path = results_dir / "class_mapping.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name"])
        for class_id, class_name in enumerate(SIGN_LABELS):
            writer.writerow([class_id, class_name])
    return path


def plot_class_distribution(class_counts: Counter, results_dir: Path) -> Path:
    path = results_dir / "class_distribution.png"
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(14, 5))
    plt.bar(classes, counts)
    plt.title("GTSRB Training Set Class Distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.xticks(classes, rotation=90)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_resolution_distribution(resolution_counts: Counter, results_dir: Path, top_n: int = 20) -> Path:
    path = results_dir / "resolution_distribution_top20.png"
    top = resolution_counts.most_common(top_n)

    labels = [f"{w}x{h}" for (w, h), _ in top]
    counts = [count for _, count in top]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), counts)
    plt.title(f"Top {top_n} Most Common Resolutions in GTSRB Train Set")
    plt.xlabel("Resolution (Width x Height)")
    plt.ylabel("Number of images")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path


def plot_example_images(records: List[Dict[str, object]], results_dir: Path) -> Path:
    path = results_dir / "sample_images_by_class.png"

    first_record_per_class: Dict[int, Dict[str, object]] = {}
    for record in records:
        class_id = int(record["class_id"])
        if class_id not in first_record_per_class:
            first_record_per_class[class_id] = record

    class_ids = sorted(first_record_per_class.keys())
    n_classes = len(class_ids)
    ncols = 8
    nrows = (n_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for i, class_id in enumerate(class_ids):
        ax = axes_flat[i]
        record = first_record_per_class[class_id]
        image = plt.imread(record["image_path"])
        ax.imshow(image)
        ax.set_title(f"{class_id}", fontsize=9)
        ax.axis("off")

    for j in range(len(class_ids), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("One Example Image per GTSRB Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_dataset_stats(records: List[Dict[str, object]], results_dir: Path) -> Path:
    path = results_dir / "dataset_stats.json"

    widths = [int(r["width"]) for r in records]
    heights = [int(r["height"]) for r in records]
    classes = [int(r["class_id"]) for r in records]

    class_counts = Counter(classes)
    resolution_counts = Counter(zip(widths, heights))

    stats = {
        "num_images": len(records),
        "num_classes": len(class_counts),
        "width": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths),
        },
        "height": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
        },
        "top_10_resolutions": [
            {"resolution": f"{w}x{h}", "count": count}
            for (w, h), count in resolution_counts.most_common(10)
        ],
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return path


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    images_dir = resolve_images_dir(args.data_root)
    records = load_annotations(images_dir)

    class_counts = Counter(int(r["class_id"]) for r in records)
    resolution_counts = Counter((int(r["width"]), int(r["height"])) for r in records)

    class_mapping_path = save_class_mapping(results_dir)
    class_dist_path = plot_class_distribution(class_counts, results_dir)
    sample_img_path = plot_example_images(records, results_dir)
    resolution_dist_path = plot_resolution_distribution(resolution_counts, results_dir)
    stats_path = save_dataset_stats(records, results_dir)

    print("GTSRB dataset analysis complete.")
    print(f"Images directory: {images_dir}")
    print(f"Total images: {len(records)}")
    print(f"Total classes: {len(class_counts)}")
    print(f"Class mapping: {class_mapping_path}")
    print(f"Class distribution plot: {class_dist_path}")
    print(f"Sample images plot: {sample_img_path}")
    print(f"Resolution distribution plot: {resolution_dist_path}")
    print(f"Dataset stats JSON: {stats_path}")


if __name__ == "__main__":
    main()
