# Baseline Model

## Overview

This report documents the implementation and evaluation of **Task 04: Baseline Model** for the GTSRB traffic sign classification project.

The objective of this task was to establish a first fully functional convolutional neural network (CNN) baseline. This baseline serves as a reproducible reference point for later experiments, including architecture changes, hyperparameter tuning, and optimization strategies in subsequent tasks.

In addition to implementing and training the model, this task also stores the most relevant training artifacts, including model checkpoints, training histories, loss curves, and offline demo outputs. These artifacts make it possible to compare future models against the initial baseline in a structured and transparent way.

---

## Baseline Architecture

The implemented neural network follows the baseline architecture defined in `TASKS.md`.

```text
Input (3 x 32 x 32)
-> Conv2d(3, 32, 3x3) + ReLU + MaxPool2d(2x2)
-> Conv2d(32, 64, 3x3) + ReLU + MaxPool2d(2x2)
-> Flatten
-> Linear(..., 256) + ReLU + Dropout(0.5)
-> Linear(256, 43)
```

The model is implemented in:

- `src/model.py`

The architecture is intentionally simple. It consists of two convolutional blocks followed by a fully connected classifier head. This keeps the model lightweight, easy to train, and suitable as a clear reference point for later comparisons.

The model outputs raw logits instead of probabilities. Therefore, no softmax activation is applied inside the `forward()` method. This is intentional because `CrossEntropyLoss` expects raw logits and internally applies log-softmax during loss computation.

---

## Training Pipeline

The training procedure is implemented in:

- `src/train.py`

The script reuses the preprocessing and data-loading pipeline developed in Task 03. This ensures that the baseline model is trained on the same standardized dataset splits and transformations, making the results reproducible and comparable across tasks.

The following configuration was used for the baseline experiments:

- Optimizer: Adam
- Learning rate: `1e-3`
- Loss function: `CrossEntropyLoss`
- Number of epochs: `10`
- Batch size: `64`
- Input image size: `32 x 32`
- Data split: Train / Validation / Test = `70 / 15 / 15`

During training, the script logs the following metrics for each epoch:

- training loss
- training accuracy
- validation loss
- validation accuracy

For each run, the following artifacts are saved:

- `models/baseline_<run_name>.pth`
- `results/baseline_loss_curve_<run_name>.png`
- `results/baseline_history_<run_name>.json`

In addition, the best-performing checkpoint is mirrored to:

- `models/baseline.pth`

This makes it easy to access the current best baseline model without having to reference a specific run name manually.

---

## Results

Two baseline training runs were executed using different random seeds. This was done to verify whether the baseline performance is stable across different initializations and data splits.

| Seed | Best Validation Accuracy | Test Accuracy | Test Loss |
|---|---:|---:|---:|
| 42  | 98.78% | 98.55% | 0.0621 |
| 123 | 99.15% | 99.29% | 0.0451 |

The corresponding training history files are stored in:

- `results/baseline_history_seed-42.json`
- `results/baseline_history_seed-123.json`

Both runs achieved very high validation and test accuracy. The difference between the two seeds is relatively small, which suggests that the model is not overly sensitive to random initialization or the specific data split used in these experiments.

The seed-123 run achieved the strongest overall performance, with a test accuracy of **99.29%** and a test loss of **0.0451**.

---

## Offline Demo Error Analysis

To further inspect the model behavior beyond aggregate metrics, the offline demo script was executed using the seed-123 baseline model and the matching split seed.

The following command was used:

```bash
.venv/bin/python src/demo_offline.py --model models/baseline_seed-123.pth --device cpu --split-seed 123
```

The output summary is stored in:

- `results/demo_offline/summary.json`

Offline demo evaluation results:

- Test samples used: `5881`
- Correct predictions: `5839`
- Incorrect predictions: `42`
- Test accuracy: `99.29%`
- Failure rate: `0.71%` (`42 / 5881`)

The following visualization artifacts were generated:

- `results/demo_offline/confusion_matrix.png`
- `results/demo_offline/misclassifications_top_confidence.png`
- `results/demo_offline/predictions_grid.png`

The confusion matrix is strongly diagonal, confirming that the model separates most traffic sign classes very well. Only a small number of samples are misclassified.

It is important to note that `misclassifications_top_confidence.png` does not display all incorrect predictions. Instead, it shows only the top-N high-confidence mistakes, with the default value being `N=20`. This visualization is useful for identifying particularly critical errors where the model was highly confident despite predicting the wrong class.

The remaining classification errors are mainly concentrated in visually similar categories, especially different speed-limit signs. Additional failure cases occur under challenging image conditions such as blur, low illumination, glare, and partial occlusion.

---

## Reproducibility Notes

Reproducibility is an important part of this task. The training script uses the `--seed` argument for both model initialization and data splitting.

For a direct comparison between a training run and the offline demo evaluation, the demo should be executed with the same `--split-seed` as the corresponding training run.

For example, when evaluating the seed-123 model, the offline demo should also use:

```bash
--split-seed 123
```

This ensures that the same test split is used and that the reported results are directly comparable to the original training run.

---

## Interpretation

The baseline CNN converges quickly and consistently across both tested seeds. Validation and test performance are high, and the gap between validation and test accuracy remains small.

This indicates that the baseline model generalizes well to unseen test data and does not show obvious signs of severe overfitting under the current setup. The small performance variation between seeds further suggests that the architecture and training pipeline are stable enough to serve as a reliable reference for future experiments.

Although the baseline is intentionally simple, it already achieves strong performance on the GTSRB dataset. Therefore, future improvements in Task 05 should focus not only on increasing accuracy, but also on evaluating robustness, reducing confident misclassifications, improving performance under difficult visual conditions, and comparing model complexity against performance gains.
