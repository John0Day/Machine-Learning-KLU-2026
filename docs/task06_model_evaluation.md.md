# Task 06 – Model Evaluation

This section presents a comprehensive evaluation of all five CNN architectures trained during
Task 05, evaluated on the held-out GTSRB test split. The evaluation covers test accuracy,
Top-5 accuracy, per-class performance, bias analysis, robustness under perturbation, overfitting
analysis, and Grad-CAM interpretability. All models were evaluated on the same test split
(15% of the full dataset, seed=42, ~5,881 images) to ensure comparability.

---

## 1. Model Comparison Overview

All five models were trained for 20 epochs using the same preprocessing pipeline, train/val/test
split, and Cross-Entropy loss function (Lecture 4). The table below summarizes the key metrics:

| Model | Params | Top-1 Acc | Top-5 Acc | Test Loss | Noisy Acc | Blurred Acc | Wrong |
|-------|--------|----------|----------|----------|-----------|------------|-------|
| Baseline CNN | 629,291 | 99.59% | 99.98% | 0.0135 | 71.52% | 97.42% | 24 |
| Deep CNN | 936,235 | 99.71% | 99.98% | 0.0094 | 70.19% | 97.84% | 17 |
| MobileNetV2 | 2,562,859 | 99.32% | 99.91% | 0.0265 | 68.24% | 97.47% | 40 |
| LeakyReLU CNN | 629,291 | 99.61% | 99.97% | 0.0125 | 60.77% | 97.64% | 23 |
| **Stride CNN** | **823,051** | **99.80%** | **100.00%** | **0.0063** | **81.94%** | **98.74%** | **12** |

> **Best model: Stride CNN** — highest Top-1 accuracy, lowest loss, fewest wrong classifications,
> and by far the best robustness under noise.

Several important observations stand out from this comparison:

**Stride CNN wins across all metrics.** By replacing MaxPool with strided convolutions (stride=2),
the model learns how to downsample the feature maps rather than using a fixed operation. As discussed
in Lecture 5, this allows the network to preserve more spatial information, which leads to better
generalization. The improvement is modest in clean accuracy (+0.21pp over Baseline) but significant
in robustness (+10.42pp under noise).

**MobileNetV2 performs worst despite having the most parameters (2.5M).** This confirms the
discussion in Lecture 5 about transfer learning limitations: MobileNetV2 was pretrained on
ImageNet at 224×224 resolution. When applied to 32×32 GTSRB images, the pretrained features
are poorly suited for the task. Fine-tuning partially compensates, but cannot fully overcome
the resolution mismatch.

**LeakyReLU CNN has the worst noise robustness (60.77%)** despite performing comparably to
Baseline on clean data. This suggests that LeakyReLU activations, while preventing dead neurons
(Lecture 4), may produce feature representations that are more sensitive to high-frequency noise.

**Deep CNN adds 307K parameters** over the Baseline but only gains +0.12pp in clean accuracy,
suggesting diminishing returns from simply adding more layers on a 32×32 input.

---

## 2. Detailed Results: Stride CNN (Best Model)

The following sections focus on the Stride CNN as the best-performing model.

### 2.1 Test Set Performance

| Metric | Value |
|--------|-------|
| Test Accuracy (Top-1) | **99.80%** |
| Test Accuracy (Top-5) | **100.00%** |
| Test Loss (Cross-Entropy) | 0.0063 |
| Wrong Classifications | 12 / 5,881 |

The Top-1 accuracy of 99.80% means the model predicted the correct traffic sign class as its
most confident prediction in all but 12 cases. The Top-5 accuracy of 100.00% is particularly
noteworthy: for every single test image, the correct label appeared among the model's five most
confident predictions. This indicates that even in the 12 error cases, the model was not
fundamentally confused — the correct answer was always within its top candidates.

The low test loss of 0.0063 confirms that the model's confidence was well-calibrated, assigning
very high probabilities to the correct class in the vast majority of cases.

### 2.2 Per-Class Analysis

Per-class accuracy was computed across all 43 traffic sign categories. The full plot is saved
at `results/task06/stride/per_class_accuracy.png`.

**5 best performing classes (100% accuracy):**
- Class 20 — *Dangerous curve right*: 100.00%
- Class 22 — *Bumpy road*: 100.00%
- Class 24 — *Road narrows on the right*: 100.00%
- Class 41 — *End of no passing*: 100.00%
- Class 42 — *End of no passing by vehicles over 3.5t*: 100.00%

**5 worst performing classes:**
- Class 30 — *Beware of ice/snow*: 98.67%
- Class 8  — *Speed limit (120km/h)*: 99.10%
- Class 3  — *Speed limit (60km/h)*: 99.11%
- Class 5  — *Speed limit (80km/h)*: 99.30%
- Class 1  — *Speed limit (30km/h)*: 99.45%

A clear pattern emerges: **almost all worst-performing classes are speed limit signs**. Speed
limit signs (30, 60, 80, 120 km/h) share the same circular red-bordered shape — the only
visual difference is the printed number inside. At 32×32 pixels, these numbers become very
similar in appearance, making them inherently harder to classify. This is a fundamental limitation
of the low input resolution, not of the model architecture itself.

Despite this challenge, even the worst class achieves 98.67% accuracy, demonstrating that
the model has successfully learned to differentiate between digit patterns inside similar-shaped signs.

### 2.3 Precision & Recall per Class

The Precision/Recall chart (`results/task06/stride/precision_recall_per_class.png`) shows
that both metrics remain above 0.98 for virtually all 43 classes. The only slight deviations
occur in the speed limit sign classes, consistent with the per-class accuracy analysis.

High precision means the model rarely predicts a class incorrectly (low false positives).
High recall means the model rarely misses a true instance of a class (low false negatives).
Both being simultaneously high confirms that the model is well-balanced and does not trade
one metric off against the other.

---

## 3. Bias Analysis

A critical aspect of evaluation is understanding whether the model performs equitably across
all classes, regardless of how frequently they appear in the training data. The GTSRB dataset
is inherently imbalanced — some sign classes appear thousands of times while others only a
few hundred times.

### Stride CNN Bias Results

| Group | Mean Accuracy |
|-------|--------------|
| Frequent classes (top 25%) | 99.78% |
| Rare classes (bottom 25%) | 100.00% |
| Accuracy gap | 0.22 percentage points |

The accuracy gap of only 0.22 percentage points confirms that the Stride CNN shows no
meaningful bias toward frequently occurring classes. The data augmentation strategy
(random rotations, color jitter, affine transforms — Lecture 4) effectively prevented the
model from over-fitting to the distribution of frequent classes.

The fact that rare classes achieve 100.00% is partly a statistical artifact: rare classes
have fewer test samples, so a small number of correct predictions yields 100%. Nevertheless,
the negligible gap confirms there is no systematic failure on underrepresented classes.

---

## 4. Overfitting Analysis (Lecture 4)

One of the central concepts from Lecture 4 is the distinction between overfitting and
underfitting. A model that overfits learns the training data too specifically — it achieves
very low training loss but fails to generalize to unseen data, resulting in a gap between
training and validation/test performance.

To assess this, we compare training loss, validation loss, and validation accuracy across
all 30 training epochs of the Baseline CNN (the reference model):

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|-------------|
| 1  | 2.3329 | 1.3194 | 57.22% |
| 5  | 0.5400 | 0.1253 | 96.46% |
| 10 | 0.3042 | 0.0357 | 99.15% |
| 15 | 0.2280 | 0.0149 | 99.61% |
| 20 | 0.1730 | 0.0115 | 99.68% |
| 25 | 0.1579 | 0.0097 | 99.63% |
| 30 | 0.1320 | 0.0039 | 99.91% |

**No significant overfitting occurred.** While training loss decreases steadily throughout
all 30 epochs, the validation loss decreases in parallel and never increases. In a classic
overfitting scenario, validation loss would begin to rise while training loss continues to
fall — a divergence that was never observed here.

**Val loss is consistently lower than train loss.** This is expected behavior when data
augmentation is applied during training but not during validation. The augmented training
data (with random rotations, color jitter, and affine transforms) is harder for the model
to fit than clean validation images. This confirms that augmentation acts as an effective
regularization technique, as described in Lecture 4.

**Small gap between val accuracy (99.91%) and test accuracy (99.80%)** confirms that the
model generalizes well to completely unseen data. A large gap here would have been a warning
sign for overfitting to the validation set — this was not the case.

**Techniques used to prevent overfitting (all from Lecture 4):**
- Dropout (p=0.23 after Optuna tuning) in the classifier head
- Batch Normalization after each convolutional block
- Data augmentation (random rotation, color jitter, affine transforms)
- Early Stopping with patience=5
- Weight decay / L2 regularization (λ=0.000333) via Adam optimizer

The combination of these techniques successfully prevented overfitting despite the model
having 823,051 trainable parameters trained on approximately 27,000 images.

---

## 5. Robustness Analysis (Lecture 4 – Generalization)

A model that performs well on clean test data is not necessarily reliable in real-world
conditions. Traffic signs can appear under adverse conditions such as sensor noise, fog,
motion blur, or dirty camera lenses.

Two perturbation types were applied to all test images:
- **Gaussian Noise** (σ=0.10): simulates sensor noise or image artifacts
- **Gaussian Blur** (k=5, σ=1.0): simulates motion blur or out-of-focus conditions

### Full Robustness Comparison

| Model | Clean | Gaussian Noise | Gaussian Blur |
|-------|-------|---------------|--------------|
| Baseline CNN | 99.59% | 71.52% | 97.42% |
| Deep CNN | 99.71% | 70.19% | 97.84% |
| MobileNetV2 | 99.32% | 68.24% | 97.47% |
| LeakyReLU CNN | 99.61% | 60.77% | 97.64% |
| **Stride CNN** | **99.80%** | **81.94%** | **98.74%** |

**Stride CNN is significantly more robust to noise than all other models (+10pp over the
next best).** This is a key advantage of strided convolutions: because the model learns
how to downsample rather than using a fixed MaxPool operation, it develops more robust
feature representations that are less sensitive to pixel-level noise.

**All models handle Gaussian blur well** (>97%), losing only 1–2 percentage points. This
is encouraging for real-world deployment, as motion blur is common in dashcam footage.
Interestingly, no blur augmentation was used during training, yet all models generalize
well — suggesting the convolutional filters respond to structural features rather than
fine pixel-level details.

**All models show significant drops under noise** (−18pp to −39pp), pointing to noise
augmentation as a clear direction for future improvement. Including noisy samples during
training would force the model to learn noise-invariant representations.

---

## 6. Grad-CAM Visualization (Lecture 5 – Interpretability)

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps highlighting the
image regions the model focuses on when making a prediction. This provides interpretability
— verifying that the model attends to the sign itself rather than background elements such
as sky, road surface, or surrounding objects.

The Grad-CAM visualizations (`results/task06/stride/gradcam_examples.png`) confirm that the
Stride CNN consistently focuses on the central region of the traffic sign, where the relevant
symbol or number is located. In the 12 misclassification cases, the heatmaps show that the
model still attends to the correct image region — the errors arise from visual ambiguity
between similar signs, not from the model focusing on irrelevant background features.

This confirms that the model has learned meaningful, interpretable representations of traffic
signs rather than exploiting spurious correlations in the training data.

---

## 7. Conclusion

The evaluation confirms that **Stride CNN is the best-performing architecture** across all
metrics — test accuracy, robustness, and loss — while using only 823K parameters, far fewer
than MobileNetV2 (2.5M).

Key findings:

- **99.80% Top-1 accuracy** and **100.00% Top-5 accuracy** on the GTSRB test set with only 12 errors
- **No overfitting**: validation loss decreased in parallel with training loss across all 30 epochs
- **No class bias**: accuracy gap between rare and frequent classes is only 0.22pp
- **Speed limit signs** are the hardest class — a fundamental consequence of 32×32 resolution
- **Noise robustness** is the main limitation (81.94%) and could be improved with noise augmentation
- **Grad-CAM** confirms the model attends to the correct sign regions, not background artifacts

The performance gap between Stride CNN and the Baseline is modest on clean data (+0.21pp),
which is expected given that GTSRB is a well-structured benchmark where even simple CNNs
achieve high accuracy with sufficient augmentation. The key differentiator is robustness:
Stride CNN's learned downsampling produces feature representations that are significantly
more resistant to real-world perturbations, making it the best choice for deployment.
