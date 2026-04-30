# CNN Traffic Sign Classification
## German Traffic Sign Recognition Benchmark (GTSRB)

**Machine Learning Project — KLU 2026**

---

## Agenda

1. Problem & Motivation
2. Dataset Overview
3. Preprocessing & Augmentation
4. Baseline CNN Architecture
5. Model Variants
6. Results & Comparison
7. Detailed Evaluation (Best Model)
8. Robustness Testing
9. Bias Analysis
10. Interpretability: Grad-CAM & t-SNE
11. Autoencoder for Anomaly Detection
12. Hyperparameter Search
13. Conclusion & Future Work

---

## 1. Problem & Motivation

**Goal:** Classify traffic signs automatically under real-world conditions

- Traffic sign recognition is a core component of driver assistance systems
- Signs must be identified under changing illumination, blur, and occlusion
- Misclassification of speed limits or stop signs has direct safety consequences

**Our approach:**
- Use the GTSRB benchmark dataset
- Train CNNs from scratch, compare architectures systematically
- Evaluate not just accuracy, but robustness, bias, and interpretability

> *Can a compact CNN trained from scratch match or exceed human-level recognition?*

---

## 2. Dataset: GTSRB

**German Traffic Sign Recognition Benchmark**

| Property | Value |
|----------|-------|
| Total images | 39,209 |
| Classes | 43 traffic sign types |
| Image size | 25×25 to 243×225 px |
| Mean image size | ~50×50 px |
| Most frequent class | Speed limit 50 km/h — 2,250 images |
| Rarest classes | Speed limit 20 km/h, Dangerous curve left, Go straight or left — 210 images each |
| Class imbalance | ~10.7× |

- All images are **pre-cropped** to the sign bounding box
- Official test set labels not available → we use an internal 15% hold-out split

![Class distribution](results/task03/class_distribution.png)

---

## 3. Data Split & Preprocessing

**Split (fixed seed 42, not stratified):**

| Split | Fraction | Images |
|-------|----------|--------|
| Training | 70% | 27,447 |
| Validation | 15% | 5,881 |
| Test | 15% | 5,881 |

**Augmentation (training only):**

| Transform | Purpose |
|-----------|---------|
| Resize 32×32 | Uniform input size |
| Random Rotation ±15° | Tilted camera angles |
| Color Jitter (brightness, contrast, saturation) | Illumination variation |
| Random Affine (translate ±10%) | Off-centre framing |
| Normalize (mean/std from GTSRB training set) | Stable gradient flow |

Validation and test receive only resize + normalize.

---

## 4. Baseline CNN Architecture

**3-block convolutional network — 629,291 parameters**

```
Input (3×32×32)
  → Block 1: Conv(32) + BN + ReLU + MaxPool  →  32×16×16
  → Block 2: Conv(64) + BN + ReLU + MaxPool  →  64×8×8
  → Block 3: Conv(128) + BN + ReLU + MaxPool →  128×4×4
  → Flatten → 2,048
  → Linear(2048→256) → ReLU → Dropout(0.5)
  → Linear(256→43)
```

**Training configuration:**
- Optimizer: Adam (lr = 1×10⁻³)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Loss: CrossEntropyLoss
- Batch size: 64 | Max epochs: 20 | Early stopping patience: 5

**Seed stability (10 epochs):**

| Seed | Val Acc | Test Acc |
|------|---------|----------|
| 42 | 98.78% | 98.55% |
| 123 | 99.15% | 99.29% |

Canonical baseline (20 epochs): **99.49% test accuracy**

---

## 5. Four Model Variants

Each variant changes **one design decision** relative to the baseline:

| Variant | Change | Motivation |
|---------|--------|------------|
| **Deep CNN** | 4th conv block (128→256), FC 256→512 | More depth for complex features |
| **MobileNetV2** | ImageNet pretrained + custom head | Transfer learning |
| **LeakyReLU CNN** | ReLU → Leaky ReLU (slope 0.01) | Avoid dead neurons |
| **Stride CNN** | MaxPool → strided conv (stride=2) | Learnable downsampling |

All trained under identical conditions: Adam, lr=1×10⁻³, 20 epochs max, same split.

---

## 6. Model Comparison Results

| Model | Test Accuracy | Wrong / 5,881 | Params | Train Time |
|-------|:---:|:---:|:---:|:---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936K** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2,563K | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

**Key findings:**
- All models exceed 99% — GTSRB is well-suited for compact CNNs
- **Deep CNN** is best: +0.32 pp over baseline, only +8.4 s training time
- **MobileNetV2** improved but costs 4× more parameters and 2× training time
- **LeakyReLU** and **Stride CNN** show no meaningful improvement

> Differences below ~0.1 pp ≈ 6 images → not statistically meaningful

![Model comparison](results/task05/model_comparison_summary.png)

---

## 7. Detailed Evaluation: Deep CNN

**Test set (5,881 images):**

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **99.81%** (11 wrong) |
| Top-5 Accuracy | **99.98%** (1 wrong) |
| Test Loss | 0.0061 |

**Five worst-performing classes:**

| Class | Accuracy | Reason |
|-------|:---:|--------|
| Pedestrians (27) | 97.62% | Similar to General caution sign |
| Bicycles crossing (29) | 97.62% | Near-identical silhouette to class 27 |
| Double curve (21) | 98.39% | Resembles single curve at 32×32 |
| Beware of ice/snow (30) | 98.67% | Snowflake detail lost at low resolution |
| Speed limit 120 km/h (8) | 99.10% | "120" vs "100" at small sizes |

All errors are **visually explainable** — not random failures.

![Confusion matrix](results/task06/deep/confusion_matrix_normalized.png)

---

## 8. Robustness Testing

The model was tested with two image distortions **at inference only** (no retraining):

| Condition | Test Accuracy | Drop |
|-----------|:---:|:---:|
| Clean | 99.81% | — |
| Gaussian Blur (kernel=5) | 97.01% | −2.80 pp |
| Gaussian Noise (σ=0.1) | **71.86%** | **−27.95 pp** |

**Conclusion:**
- Blur: model handles it well — does not fully rely on sharp edges
- **Noise: severe drop** — model was never trained on noisy inputs (distribution shift)
- Clean benchmark accuracy ≠ real-world robustness

> Most important next step: add noise augmentation during training

---

## 9. Class Frequency Bias Analysis

**Question:** Does the model perform worse on rare classes?

| Group | Avg. training images | Mean Test Accuracy |
|-------|:---:|:---:|
| 10 most frequent classes | ~1,374 | **99.87%** |
| 10 rarest classes | ~169 | **99.52%** |
| Gap | — | **0.34 pp** |

- Only **0.34 pp** gap despite ~11× class imbalance
- Several rare classes (Speed limit 20 km/h, Dangerous curve left) reach **100%**
- Augmentation helped rare classes generalise effectively

**Caution:** Based on one training run and one split — rare classes have few test images, so one mistake changes accuracy significantly.

![Bias analysis](results/task06/deep/bias_analysis_mean_accuracy.png)

---

## 10. Interpretability

### Grad-CAM
Highlights which image regions drove each prediction:

- Model focuses on **sign shape, symbol, and colour**
- Not on background (sky, road, other objects)
- Errors occur mainly in **visually ambiguous regions** at 32×32 resolution

![Grad-CAM examples](results/task06/deep/gradcam_examples.png)

### t-SNE Latent Space
2,000 validation samples projected from 512D → 2D:

- Most of the 43 classes form **distinct clusters**
- Overlap mainly between speed limit signs (same shape, different numeral) and similar warning triangles
- Consistent with the per-class error analysis

![t-SNE projection](results/tsne_feature_space.png)

---

## 11. Autoencoder for Anomaly Detection

**Problem:** A classifier always assigns a class — even for unknown inputs.

**Solution:** Train a convolutional autoencoder as an auxiliary anomaly detector.

- Encoder: 3×32×32 → 3 conv blocks → **128-dim latent vector**
- Decoder: mirrors encoder with transposed convolutions
- Loss: Mean Squared Error (MSE)
- Threshold: **95th percentile** of validation reconstruction errors

| Metric | Value |
|--------|-------|
| Final val loss | 0.5118 |
| Anomaly threshold | 1.091 |
| Flagged (validation set) | 294 / 5,881 (5.0%) |

- In-distribution signs: low reconstruction error
- Potential anomalies: high error → long right tail in error distribution
- **Proof of concept** — no true out-of-distribution test set available

---

## 12. Hyperparameter Search (Optuna)

**Goal:** Verify that our manually chosen hyperparameters are in a robust region.

- Method: Bayesian optimisation (Tree-structured Parzen Estimator)
- Architecture: Stride CNN | 30 trials × 10 epochs each

| Hyperparameter | Search Range | Best Value |
|---------------|-------------|:---:|
| Learning rate | 1×10⁻⁴ to 1×10⁻² | **1.24×10⁻³** |
| Dropout | 0.2 – 0.6 | **0.274** |
| Batch size | 32 / 64 / 128 | **32** |
| Optimizer | Adam / SGD | **Adam** |
| Weight decay | 1×10⁻⁵ to 1×10⁻³ | **6.98×10⁻⁴** |

**Best trial: 99.91% validation accuracy**

**Key takeaways:**
- Adam consistently outperformed SGD in all top-5 trials
- Default lr=1×10⁻³ is close to the best found (1.24×10⁻³) ✓
- Optimal dropout (0.274) lower than our default (0.5) — Stride CNN needs less regularisation

---

## 13. Conclusion & Future Work

### Three main findings:

**1. Compact CNNs trained from scratch are sufficient**
Baseline: 99.49% — comparable to the 98.84% human recognition rate on the official benchmark.

**2. Depth was the only impactful architectural change**
Deep CNN: 99.81%, reduces errors from 30 → 11. Leaky ReLU, strided convolutions: no benefit.
MobileNetV2: improved accuracy but 4× parameters and 2× training time — no efficiency advantage.

**3. Clean accuracy ≠ robustness**
−27.95 pp under Gaussian noise. The main remaining challenge is distribution shift, not clean-image classification.

---

### Future Work

| Priority | Next step |
|----------|-----------|
| High | Add noise & blur augmentation during training |
| High | Repeat comparison with multiple seeds and splits |
| Medium | Increase input resolution to 64×64 (especially for similar signs) |
| Low | Add object detection stage for full road images |
| Low | Test on non-German traffic sign datasets |

---

*Thank you — Questions?*
