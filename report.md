# CNN Traffic Sign Classification — Final Report
**German Traffic Sign Recognition Benchmark (GTSRB)**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset](#2-dataset)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Baseline Model](#4-baseline-model)
5. [Model Improvements](#5-model-improvements)
6. [Model Evaluation](#6-model-evaluation)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

Traffic sign recognition is a core component of modern advanced driver assistance systems (ADAS) and autonomous vehicles. A reliable classifier must handle a wide variety of visual conditions — varying illumination, partial occlusion, motion blur, and significant differences in sign size — while distinguishing between 43 distinct traffic sign categories with low error tolerance.

This project implements and evaluates a series of convolutional neural network (CNN) architectures for traffic sign classification on the German Traffic Sign Recognition Benchmark (GTSRB). The work is structured in six stages: dataset analysis, preprocessing, baseline modeling, architecture improvements, thorough evaluation, and this report synthesizing all findings.

The central research questions are:

- Can a compact, from-scratch CNN reach near-human accuracy on GTSRB?
- Do architectural improvements (depth, activation functions, strided convolutions) meaningfully outperform the baseline?
- Does transfer learning from ImageNet provide an advantage on this specialized dataset?
- How robust is the best model to real-world image degradations such as noise and blur?
- Does the model treat rare and frequent traffic sign classes equally well?

---

## 2. Dataset

### 2.1 Overview

The GTSRB dataset was recorded from a car-mounted camera on German roads. It contains **39,209 training images** across **43 traffic sign classes**. Images are provided in PPM format at varying resolutions, ranging from as small as 15×15 pixels to over 250×250 pixels. This variability reflects real-world conditions where a sign may appear very small in the distance or large and close-up.

### 2.2 Class Distribution

![Class distribution across all 43 GTSRB traffic sign categories](results/task03/class_distribution.png)

The dataset is **not uniformly distributed**. The most frequent classes (e.g. Speed limit 30km/h with 1,552 training images) have roughly ten times as many samples as the rarest classes (e.g. Speed limit 20km/h with only 140 images). This class imbalance is a central concern for both training and evaluation, as a model could achieve high average accuracy simply by performing well on frequent classes while failing on rare ones.

| Metric | Value |
|--------|-------|
| Total training images | 39,209 |
| Number of classes | 43 |
| Most frequent class | Speed limit (30km/h) — 1,552 images |
| Least frequent class | Speed limit (20km/h) — 140 images |
| Imbalance ratio (max/min) | ~11× |

### 2.3 Sample Images

![One representative image per class](results/task03/sample_images_by_class.png)

The sample grid illustrates the visual diversity of the dataset. Even within a single class, images vary substantially in brightness, contrast, viewing angle, and background. This makes the classification task non-trivial and motivates careful preprocessing and augmentation.

---

## 3. Data Preprocessing

### 3.1 Data Split

The 39,209 training images are divided into three non-overlapping subsets using a fixed random seed (42) for reproducibility:

| Split | Fraction | Images |
|-------|----------|--------|
| Training | 70% | 27,447 |
| Validation | 15% | 5,881 |
| Test | 15% | 5,881 |

The validation set is used during training to monitor generalization and apply early stopping. The test set is held out entirely and evaluated only once per model — evaluating on data that was seen during training or hyperparameter selection would produce overly optimistic results and undermine the trustworthiness of the reported metrics. Using a fixed seed ensures that all model variants are evaluated on identical splits, making comparisons fair.

### 3.2 Image Transformations

All images are resized to **32×32 pixels** before processing. This uniform size is required because the fully connected classifier layers have fixed weight matrix dimensions. At 32×32 the images are compact enough for fast training while retaining enough detail for the model to distinguish sign shapes, symbols, and colors.

**Training transforms** apply stochastic augmentations that are different each time:

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| Random Rotation | ±15° | Simulates tilted camera angles |
| Color Jitter | brightness ±0.4, contrast ±0.4, saturation ±0.3 | Simulates lighting and weather variation |
| Random Affine | translate ±10% | Simulates off-center sign placement |
| Normalize | mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629) | Centers input distribution |

**Validation and test transforms** are fully deterministic — only resize, convert to tensor, and normalize. No augmentation is applied during evaluation, so that measured accuracy honestly reflects model performance on unmodified inputs.

### 3.3 Normalization

Pixel values are converted from the range [0, 255] to floating-point [0.0, 1.0], then normalized per channel using the mean and standard deviation computed from the GTSRB training set. Normalization is essential for stable optimization: without it, large differences in pixel scales across channels can distort the loss surface and slow down or destabilize convergence.

### 3.4 Data Augmentation as Regularization

Augmentation artificially increases the effective diversity of the training set. The model never sees the exact same pixel values twice, which prevents it from memorizing specific training examples. This is particularly important for the rarest sign categories that have fewer than 200 training samples and would otherwise be prone to overfitting.

### 3.5 Mini-Batch Loading and Early Stopping

Images are fed to the model in mini-batches of size 64. Mini-batch stochastic gradient descent introduces noise into the gradient estimates, which helps the optimizer escape poor local minima. The training DataLoader uses shuffling so that every mini-batch sees a fresh random sample each epoch.

Early stopping with patience 5 halts training when validation accuracy does not improve for five consecutive epochs. The checkpoint with the highest validation accuracy — not the last epoch — is restored for final evaluation. This prevents overfitting to the training set in later epochs and ensures that reported results reflect the model's best generalization, not its final state.

---

## 4. Baseline Model

### 4.1 Architecture

The baseline CNN consists of three convolutional blocks followed by a fully connected classifier. The architecture diagram below (left) shows the full layer sequence with feature map dimensions at each stage.

![Architecture comparison: Baseline CNN (left) vs. Deep CNN (right)](results/diagrams/architecture_comparison.png)

**Total trainable parameters: 629,291**

Each convolutional block uses padding=1 to preserve spatial dimensions before pooling, BatchNorm to stabilize gradient flow, and ReLU as the activation function. Dropout(0.5) in the classifier regularizes the network by randomly dropping half the hidden units during training, preventing co-adaptation of neurons. The model outputs raw logits — no softmax is applied — because the CrossEntropyLoss function applies log-softmax internally, which is numerically more stable.

### 4.2 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss function | CrossEntropyLoss |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| Input size | 32×32 |

Adam is chosen over plain SGD because it adapts the learning rate individually for each parameter, which typically leads to faster and more stable convergence. The ReduceLROnPlateau scheduler halves the learning rate whenever validation loss stops improving for three epochs, allowing fine-grained adjustments as training matures.

### 4.3 Results

Two runs were conducted with different random seeds to verify stability:

| Seed | Best Val Accuracy | Test Accuracy | Test Loss |
|------|------------------|--------------|-----------|
| 42   | 98.78%           | 98.55%       | 0.0621    |
| 123  | 99.15%           | 99.29%       | 0.0451    |

The performance is consistent across both seeds, confirming that the architecture and training pipeline are stable. The small difference between runs is attributable to random weight initialization and mini-batch ordering rather than structural instability.

![Baseline training curves (seed 42): training and validation loss and accuracy over epochs](results/task04/baseline_loss_curve_seed-42.png)

The loss curves show smooth convergence with no signs of severe overfitting — the gap between training and validation accuracy remains small throughout training, and early stopping engages after the validation plateau is reached.

---

## 5. Model Improvements

### 5.1 Overview

Four architectural variants were trained under identical conditions to enable a fair comparison. All models were trained for up to 20 epochs with the same optimizer, scheduler, and early stopping configuration.

| Model | Test Accuracy | Wrong / 5881 | Parameters | Training Time |
|-------|-------------|:---:|-----------|:---:|
| Baseline CNN | 99.49% | 30 | 629,291 | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936,235** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2,562,859 | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629,291 | 271.5 s |
| Stride CNN | 99.52% | 28 | 823,051 | 236.9 s |

![Model comparison summary: test accuracy, accuracy vs. parameters, and training time](results/task05/model_comparison_summary.png)

![Training accuracy curves for all five model variants](results/task05/model_comparison_curves.png)

### 5.2 Variant A — Deep CNN

The Deep CNN adds a fourth convolutional block (128→256 filters) and expands the classifier from 256 to 512 hidden units, as shown in the architecture diagram in Section 4.1 (right side). Additional depth allows the network to learn increasingly abstract representations: early layers detect edges and gradients, deeper layers combine these into shape- and symbol-level features. This variant achieves the **highest test accuracy of 99.81%** — only 11 wrong predictions out of 5,881 — with only a 49% parameter increase over the baseline. Training time is nearly identical (284 s vs. 276 s), making it the most cost-effective improvement overall.

### 5.3 Variant B — MobileNetV2 (Transfer Learning)

MobileNetV2, pretrained on ImageNet (1.2 million images, 1,000 classes), was used as a feature extractor with a custom two-layer classifier head adapted for the 43 GTSRB classes. All weights including the backbone were fine-tuned during training.

Transfer learning is motivated by the fact that low-level visual features — edges, textures, gradients — are shared across many image domains. The pretrained backbone provides a strong initialization, particularly beneficial for the rarest GTSRB classes with fewer than 200 training samples.

MobileNetV2 achieves 99.66% test accuracy but requires **4× more parameters** (2.56M vs. 629K) and nearly **twice the training time** (519 s vs. 276 s) compared to the baseline for only a 0.17 percentage point gain. On this dataset, transfer learning does not justify its additional computational cost.

### 5.4 Variant C — LeakyReLU CNN

This variant replaces all ReLU activations with Leaky ReLU (negative slope = 0.01). Standard ReLU can produce "dead neurons" — units whose input is persistently negative, causing the gradient to be exactly zero and the neuron to permanently stop learning. Leaky ReLU prevents this by maintaining a small gradient (0.01×z) for negative inputs.

Despite this theoretical advantage, Leaky ReLU CNN achieves 99.46% — marginally below the baseline (99.49%). This suggests that with BatchNorm stabilizing the activations throughout the network, dead neurons are not a significant problem at this scale.

### 5.5 Variant D — Stride CNN

Instead of fixed MaxPool layers, the Stride CNN uses strided convolutions (stride=2) for downsampling. While MaxPool selects the maximum value in each 2×2 region by a fixed rule, strided convolutions learn how to downsample, potentially preserving more useful spatial information.

The Stride CNN achieves 99.52% test accuracy and is the **fastest to train** (236.9 s), making it attractive when computational budget is limited. The accuracy difference compared to the baseline is marginal (0.03 pp).

---

## 6. Model Evaluation

The Deep CNN was selected as the best model and evaluated in depth on the held-out test set.

### 6.1 Test Set Performance

| Metric | Value |
|--------|-------|
| Test Accuracy (Top-1) | **99.81%** |
| Test Accuracy (Top-5) | **99.98%** |
| Test Loss | 0.0061 |
| Wrong Classifications | 11 / 5,881 |

The Top-5 accuracy of 99.98% means that in all but two test cases, the correct class appears among the model's five most confident predictions.

### 6.2 Confusion Matrix

![Normalized confusion matrix of the Deep CNN on the test set](results/task06/deep/confusion_matrix_normalized.png)

The confusion matrix is strongly diagonal, confirming that the model correctly classifies the vast majority of test samples. The few off-diagonal entries are concentrated among visually similar sign pairs — different speed limit signs and warning signs with comparable layouts.

### 6.3 Per-Class Accuracy

![Per-class test accuracy across all 43 GTSRB classes](results/task06/deep/per_class_accuracy.png)

**Five best-performing classes (100% accuracy):** Stop, Dangerous curve left, Dangerous curve right, End of no passing, End of no passing by vehicles over 3.5t.

**Five worst-performing classes:**

| Class | Name | Test Accuracy |
|-------|------|:---:|
| 27 | Pedestrians | 97.62% |
| 29 | Bicycles crossing | 97.62% |
| 21 | Double curve | 98.39% |
| 30 | Beware of ice/snow | 98.67% |
| 8  | Speed limit (120km/h) | 99.10% |

The worst-performing classes share a common characteristic: they are visually similar to other classes. Pedestrian and bicycle crossing signs have comparable layouts with subtle icon differences that are difficult to resolve at 32×32 pixel resolution.

### 6.4 Precision and Recall

![Precision and recall per class for the Deep CNN](results/task06/deep/precision_recall_per_class.png)

Precision and recall are consistently high across all 43 classes, with no systematic failure mode in any particular category. The few classes with slightly reduced scores correspond exactly to the visually ambiguous categories identified in the per-class accuracy analysis.

### 6.5 Misclassified Examples

![High-confidence misclassifications: cases where the model was wrong but confident](results/task06/deep/misclassifications_top_confidence.png)

The misclassification grid shows the 11 test images the model predicted incorrectly. In most cases the error is understandable: the image quality is degraded, the sign is partially occluded, or the sign closely resembles another class. This confirms that remaining errors are concentrated in genuinely hard cases rather than systematic misclassifications of an entire category.

### 6.6 Bias Analysis

A critical concern for real-world deployment is whether the model performs disproportionately worse on underrepresented classes. Fairness across class frequencies is evaluated by splitting the 43 classes into the 10 most frequent and 10 least frequent by training count.

![Mean test accuracy for frequent vs. rare traffic sign classes](results/task06/deep/bias_analysis_mean_accuracy.png)

| Group | Training images (avg.) | Mean Test Accuracy |
|-------|----------------------|-------------------|
| Frequent classes (top 10) | ~1,374 per class | 99.87% |
| Rare classes (bottom 10) | ~169 per class | 99.52% |
| Gap | — | **0.34 percentage points** |

The accuracy gap of only **0.34 pp** between the most and least represented classes demonstrates that the pipeline handles class imbalance effectively. Notably, several of the rarest classes — Speed limit (20km/h) with only 140 training images, Dangerous curve left with 145 — achieve 100% test accuracy. This result validates that the data augmentation strategy and training procedure generalize well even under significant class imbalance.

This is a meaningful result for trustworthiness: a model that is accurate on average but fails systematically on rare classes would be unsuitable for deployment, as rare signs (e.g. road narrows, bicycles crossing) are precisely those that require reliable recognition in safety-critical situations.

### 6.7 Robustness Testing

Real-world deployment involves conditions not present in clean test data. Two standard perturbations were evaluated:

| Condition | Test Accuracy | Δ vs. Clean |
|-----------|-------------|:-----------:|
| Clean | 99.81% | — |
| Gaussian Blur (kernel=5) | 97.01% | −2.80 pp |
| Gaussian Noise (σ=0.1) | 71.86% | **−27.95 pp** |

The model maintains strong performance under blur, which simulates motion blur or out-of-focus optics — conditions that commonly occur in moving vehicles. However, **Gaussian noise causes a dramatic accuracy drop to 71.86%**. This is a well-known vulnerability of CNNs trained exclusively on clean images: pixel-level noise that humans largely ignore can fundamentally alter the activation patterns in a convolutional network.

This finding represents the most significant limitation of the current system and must be considered when assessing suitability for real deployment in environments with low-quality camera sensors.

### 6.8 Grad-CAM Interpretability

![Grad-CAM visualizations: image regions that most influenced the model's predictions](results/task06/deep/gradcam_examples.png)

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the image regions that most strongly influenced the model's predictions by computing the gradient of the predicted class score with respect to the final convolutional feature maps.

The visualizations confirm that the model attends to the relevant sign regions — the shape, central symbol, and color — rather than background artifacts such as the sky, road surface, or surrounding vehicles. This is an important trustworthiness indicator: a model achieving high accuracy by exploiting spurious background correlations rather than the sign itself would be fragile under distribution shift. The Grad-CAM results provide evidence that this is not the case here.

---

## 7. Discussion

### 7.1 Summary of Findings

All five models exceed 99% test accuracy on the clean GTSRB test split, confirming that CNN-based classifiers are well-suited to this task. The key findings are:

**Depth helps, but with diminishing returns.** Adding a fourth convolutional block (Deep CNN) increases accuracy by 0.32 pp over the baseline at minimal additional cost. This is the most effective improvement found.

**Transfer learning is not necessary for GTSRB.** MobileNetV2 delivers smaller accuracy gains at significantly higher computational cost. The GTSRB training set (27,447 images) is large enough for from-scratch CNNs to learn excellent representations without ImageNet pretraining.

**Activation function and downsampling choices have minimal impact.** Replacing ReLU with Leaky ReLU or MaxPool with strided convolutions produces no meaningful accuracy change, suggesting that BatchNorm is the dominant stabilizing factor, and that the choice of downsampling method is secondary.

**The model generalizes well across rare classes.** The 0.34 pp accuracy gap between frequent and rare classes confirms that the augmentation and training strategy mitigate class imbalance effectively, without requiring explicit reweighting or oversampling.

**Noise robustness is the main open challenge.** The 27.95 pp accuracy drop under Gaussian noise is the clearest limitation. This points to a concrete gap between benchmark performance and real-world reliability.

### 7.2 Assumptions and Limitations

**Fixed 32×32 resolution.** Downsampling all images to 32×32 makes the pipeline fast and lightweight but discards spatial detail. Some visually similar classes (e.g. pedestrian vs. bicycle crossing signs) might be more reliably distinguished at higher resolution (64×64 or 96×96), at the cost of larger models and longer training.

**Single random seed for improved models.** While two seeds were compared for the baseline, the improved model variants were each trained once. Performance estimates would be statistically more reliable with multiple independent runs.

**Clean training data.** No noise or blur augmentation was applied during training. The model was optimized for clean inputs only, which directly explains its poor noise robustness. A production system would require noise injection during training or a preprocessing denoising step.

**Fixed data split.** The 70/15/15 split is applied once with a fixed seed. Cross-validation would provide a more robust estimate of generalization performance but was not applied here due to the computational cost of training five model variants.

**Benchmark vs. deployment gap.** The GTSRB test set shares the same distribution as the training data. In real deployment, signs may appear under conditions not represented in the dataset — unusual weather, different countries, damaged or vandalized signs. Performance under distribution shift was not evaluated beyond the noise and blur robustness tests.

### 7.3 Suitability Assessment

For the purpose of this course project — demonstrating CNN-based traffic sign classification on a standard benchmark — the approach is fully suitable. The Deep CNN achieves near-perfect accuracy (99.81%), generalizes well across class frequencies, and the Grad-CAM analysis confirms it learns meaningful visual features.

For real-world deployment in a safety-critical system, the noise sensitivity (71.86% accuracy under σ=0.1 Gaussian noise) would need to be addressed before the system could be considered reliable. The most practical path forward would be augmenting the training set with noise and blur perturbations, which is a well-established technique for improving CNN robustness at minimal cost.

---

## 8. Conclusion

This project demonstrates that a compact from-scratch CNN can achieve near-perfect accuracy on the GTSRB traffic sign classification benchmark. The Deep CNN — a four-block convolutional network with 936,235 trainable parameters — reaches **99.81% top-1 test accuracy**, misclassifying only 11 out of 5,881 test images.

The systematic comparison of five model variants shows that architectural depth is the most cost-effective improvement, while transfer learning offers diminishing returns on a dataset of this size. The bias analysis confirms that the pipeline handles class imbalance well, with only a 0.34 pp accuracy gap between the most and least frequent sign categories — a meaningful result for deployment trustworthiness. Grad-CAM visualizations confirm that the model bases its predictions on the sign itself rather than spurious background correlations.

The primary identified limitation is noise sensitivity: a 27.95 pp accuracy drop under Gaussian noise represents the most important gap between benchmark performance and real-world reliability. Targeted noise augmentation during training is the clearest path to closing this gap and constitutes the most valuable direction for future work.

---

## Appendix: Generated Artifacts

| Task | Key Output Files |
|------|-----------------|
| Task 02 | `results/class_mapping.csv`, `results/task03/class_distribution.png` |
| Task 03 | `results/preprocessing_stats.json`, `results/preprocessing_sample_grid.png` |
| Task 04 | `models/baseline.pth`, `results/task04/baseline_history_seed-42.json`, `results/task04/baseline_loss_curve_seed-42.png` |
| Task 05 | `models/deep_cnn.pth`, `results/task05/model_comparison.json`, `results/task05/model_comparison_summary.png` |
| Task 06 | `results/task06/deep/evaluation_summary.json`, `results/task06/deep/gradcam_examples.png`, `results/task06/deep/confusion_matrix_normalized.png`, `results/task06/deep/bias_analysis_mean_accuracy.png` |
