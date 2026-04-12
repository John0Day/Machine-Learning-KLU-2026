# Baseline Model

## Overview

This report documents **Task 04 (Baseline Model)** for the GTSRB traffic sign project. The goal is to implement a first working CNN baseline, train it reproducibly, and store core training artifacts for later comparison.

---

## Baseline Architecture

The implemented model follows the baseline structure from `TASKS.md`:

```text
Input (3 x 32 x 32)
-> Conv2d(3, 32, 3x3) + ReLU + MaxPool2d(2x2)
-> Conv2d(32, 64, 3x3) + ReLU + MaxPool2d(2x2)
-> Flatten
-> Linear(..., 256) + ReLU + Dropout(0.5)
-> Linear(256, 43)
```

Implementation file:

- `src/model.py`

Note: the model outputs raw logits. Softmax is not applied in `forward()` because `CrossEntropyLoss` internally applies log-softmax.

---

## Training Pipeline

Training is implemented in `src/train.py` and reuses the Task-03 preprocessing/data-loader pipeline.

Configuration used for baseline runs:

- Optimizer: Adam (`lr=1e-3`)
- Loss: CrossEntropyLoss
- Epochs: 10
- Batch size: 64
- Input size: 32 x 32
- Data split: Train/Val/Test = 70/15/15 (from Task 03)

Per epoch, the script logs:

- train loss and train accuracy
- validation loss and validation accuracy

Saved outputs per run:

- `models/baseline_<run_name>.pth`
- `results/baseline_loss_curve_<run_name>.png`
- `results/baseline_history_<run_name>.json`

Additionally, the best checkpoint is mirrored to:

- `models/baseline.pth`

---

## Results

Two baseline runs were executed with different seeds.

| Seed | Best Val Accuracy | Test Accuracy | Test Loss |
|---|---:|---:|---:|
| 42  | 98.78% | 98.55% | 0.0621 |
| 123 | 99.15% | 99.29% | 0.0451 |

Source files:

- `results/baseline_history_seed-42.json`
- `results/baseline_history_seed-123.json`

---

## Offline Demo Error Analysis (No Camera)

To visualize predictions and failure modes, the offline demo was executed with the seed-123 model and matching split seed:

```bash
.venv/bin/python src/demo_offline.py --model models/baseline_seed-123.pth --device cpu --split-seed 123
```

Output summary (`results/demo_offline/summary.json`):

- Test samples used: `5881`
- Correct predictions: `5839`
- Wrong predictions: `42`
- Test accuracy: `99.29%`
- Fail rate: `0.71%` (`42 / 5881`)

Generated artifacts:

- `results/demo_offline/confusion_matrix.png`
- `results/demo_offline/misclassifications_top_confidence.png`
- `results/demo_offline/predictions_grid.png`

Interpretation:

- The confusion matrix is strongly diagonal, confirming high overall class separability.
- `misclassifications_top_confidence.png` does **not** show all failures; it shows only the top-N high-confidence mistakes (default `N=20`).
- The remaining errors are concentrated in visually similar classes (especially speed-limit signs) and difficult image conditions (blur, low light, glare, partial occlusion).

Note on reproducibility:

- `train.py` uses `--seed` for both initialization and data split.
- For direct comparison with a training run, the demo should use the same `--split-seed` as the training seed.

---

## Interpretation

The baseline converges quickly and stably across both seeds. Validation and test metrics are consistently high, and the spread between seeds is small, indicating that the architecture and training setup are robust enough to serve as a strong reference point for Task 05 model-improvement experiments.

---

## Summary

Task 04 is functionally complete:

- baseline CNN implemented (`src/model.py`)
- training loop with validation logging implemented (`src/train.py`)
- loss curves generated
- best model weights saved
- reproducible multi-seed run artifacts recorded

This establishes a solid baseline for later architecture and optimization comparisons.
