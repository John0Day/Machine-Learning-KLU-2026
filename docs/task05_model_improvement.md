# Model Improvement (Task 05)

## Overview

To assess whether the baseline CNN can be further improved, four additional model variants were trained and compared against the baseline. Two additional techniques — latent space visualisation and anomaly detection via autoencoders — were also applied. The comparison covers test accuracy, validation accuracy, number of trainable parameters, and training time.

---

## Variant A – Deep CNN

The first variant extends the baseline architecture by adding a fourth convolutional block. The number of filters increases progressively (32 → 64 → 128 → 256), allowing the network to learn increasingly abstract and high-level features. The classifier head is also expanded from 256 to 512 hidden units to accommodate the larger feature representation.

The motivation for this variant is to test whether additional depth improves performance. Deeper networks can capture more complex patterns, but they also require more parameters and training time, and risk overfitting if the dataset is too small relative to model capacity.

---

## Variant B – MobileNetV2 (Transfer Learning)

The second variant uses MobileNetV2, a lightweight convolutional neural network pretrained on ImageNet (1.2 million images, 1000 classes). The pretrained feature extractor is kept and a custom classifier head is attached, consisting of two linear layers with Dropout, adapted for the 43 GTSRB classes.

Transfer learning is motivated by the observation that low-level visual features such as edges, textures, and shapes are shared across many image classification tasks. By starting from weights already tuned on a large dataset, the model can achieve strong performance even with relatively little task-specific training data. This is particularly relevant for underrepresented classes in GTSRB where fewer training samples are available.

During training, all weights including the backbone are fine-tuned (full fine-tuning), allowing the pretrained features to be adapted to the specific appearance of traffic signs.

---

## Variant C – Leaky ReLU CNN

The third variant replaces ReLU with Leaky ReLU throughout the network. Lecture 4 identifies the "dead neuron" problem with standard ReLU: when the input to a ReLU neuron is consistently negative, it outputs zero for all inputs and its gradient becomes zero, causing the neuron to stop learning permanently. Leaky ReLU addresses this by allowing a small gradient (0.01×z) for negative inputs, keeping all neurons active during training.

## Variant D – Stride CNN

The fourth variant replaces MaxPool layers with strided convolutions (stride=2). Lecture 5 explains striding as an alternative downsampling strategy: instead of a fixed operation that selects the maximum value in each region, strided convolutions learn how to downsample the feature maps. This gives the network more flexibility in deciding which information to retain, potentially preserving more useful spatial detail.

## Latent Space Visualisation (t-SNE)

Inspired by Lecture 7's discussion of latent space visualisation with autoencoders and dimensionality reduction techniques, we extract the feature vectors from the trained baseline CNN (the output of the penultimate layer, before the final classifier) and project them to two dimensions using t-SNE. This reveals the structure of the learned representation: classes that are visually similar (e.g. different speed limit signs) should cluster together, while visually distinct classes should be well-separated.

## Anomaly Detection via Autoencoder

Following the anomaly detection approach from Lecture 7, a convolutional autoencoder is trained on the GTSRB training set. The autoencoder learns to reconstruct known traffic signs with low mean squared error (MSE). When presented with an unknown or damaged sign, the reconstruction error is significantly higher, allowing the model to flag such images as anomalies. The anomaly threshold is set at the 95th percentile of reconstruction errors on the validation set.

This approach demonstrates an unsupervised application of deep learning that complements the supervised CNN classifier: the CNN classifies known signs, while the autoencoder detects signs that fall outside the training distribution.

## Training Setup

All three models were trained under identical conditions to ensure a fair comparison:

- Optimizer: Adam (lr = 0.001)
- Loss: Cross-Entropy
- Learning rate schedule: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience = 5 epochs
- Batch size: 64
- Image size: 32×32
- Data split: 70% train / 15% val / 15% test (same split as baseline)

---

## Results

The results below are filled in after training. Refer to `results/model_comparison.md` for the automatically generated table.

| Model | Test Accuracy | Val Accuracy | Parameters | Training Time |
|---|---|---|---|---|
| Baseline CNN | 99.78% | 99.91% | 629,291 | — |
| Deep CNN | — | — | — | — |
| MobileNetV2 | — | — | — | — |
| LeakyReLU CNN | — | — | — | — |
| Stride CNN | — | — | — | — |

*Results to be filled in after training. See `results/model_comparison.md` for the auto-generated table.*

---

## Discussion

The baseline CNN already achieves 99.78% test accuracy, which leaves limited room for improvement on this dataset. The comparison with the Deep CNN variant reveals whether additional depth provides measurable benefit given the already high baseline performance. The MobileNetV2 variant demonstrates the practical value of transfer learning: despite being designed for a general image classification task, the pretrained weights provide a strong initialization that can be efficiently adapted to traffic sign classification.

A key observation for the report is the trade-off between model complexity and accuracy: a larger model is not always better, especially when the simpler model already saturates the available performance on the dataset.

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
