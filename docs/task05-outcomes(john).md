# Task 05 Outcomes (Beginner-Friendly)

## What This Document Is

This file explains the results of Task 05.
We trained several image-classification models and compared them to see which one recognizes traffic signs best.

Run command:

```bash
python src/train_improved.py --epochs 20 --batch-size 64 --device mps
```

Alternative command (works on any machine):

```bash
python src/train_improved.py --epochs 20 --batch-size 64 --device cpu
```

Main result file:

- `results/model_comparison.json`

## Device Choice: `mps` vs `cpu`

You can run Task 05 on different compute devices:

- `--device mps`: uses Apple Metal GPU acceleration (Mac with Apple Silicon).
- `--device cpu`: uses the processor only (works everywhere, usually slower).

What changes when you switch device:

- Training speed:
  - `mps` is typically much faster.
  - `cpu` is typically slower, especially for larger models.
- Final accuracy:
  - Usually very similar.
  - Small differences can happen because deep learning training is not always bit-identical across hardware backends.
- Reproducibility:
  - For strict comparisons, use the same device, same hyperparameters, and same random seed setup.

Practical rule:

- Use `mps` for normal development on your Mac (faster iterations).
- Use `cpu` if `mps` is unavailable or if you need a fallback that works on almost any system.

## What The Models Actually Do

Each model looks at a traffic sign image and answers:
"Which of the 43 sign classes is this?"

For every image, the dataset already contains the correct answer (the class label).
During training, the model compares its guess with the correct label and adjusts itself to make fewer mistakes over time.

This is called **supervised learning**.

Note: The autoencoder/anomaly part in Task 05 is different. It is **unsupervised** and learns to reconstruct images instead of predicting class labels.

## How The Data Was Split

Total images: `39,209`

- Train set (70%): `27,447` images
- Validation set (15%): `5,881` images
- Test set (15%): `5,881` images

Why this matters:

- Train set: used to learn.
- Validation set: used during training to monitor progress and pick the best checkpoint.
- Test set: used only at the end for final, fair evaluation on unseen images.

## Results In Plain Language

| Model | Test Accuracy | Approx. Wrong Predictions (out of 5,881) | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629,291 | 275.6s |
| Deep CNN | **99.81%** | **11** | 936,235 | 284.0s |
| MobileNetV2 | 99.66% | 20 | 2,562,859 | 518.7s |
| LeakyReLU CNN | 99.46% | 32 | 629,291 | 271.5s |
| Stride CNN | 99.52% | 28 | 823,051 | 236.9s |

How to read this:

- Accuracy tells you the percentage of correct predictions on unseen test data.
- "Wrong predictions" is often easier to understand than percentages.
- Parameters roughly indicate model size/complexity.
- Training time tells you how long the model needed to train.

## Which Model Is Best?

For this run, the best model is **Deep CNN**.

Why:

- It has the highest test accuracy (`99.81%`).
- It has the fewest estimated mistakes on the test set (`11` wrong out of `5,881`).

## Is The Baseline Model Still Good?

Yes. The baseline is already very strong:

- `99.49%` test accuracy
- about `30` wrong predictions out of `5,881`

So Task 05 improves from "already very good" to "even better."

## Final Recommendation

- If your goal is the best possible accuracy: use **Deep CNN**.
- If your goal is faster training with still very strong performance: use **Stride CNN**.
- If you need a simple and reliable reference model: keep **Baseline CNN** as your benchmark.
