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
8. [Future Work](#8-future-work)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

### 1.1 Problem Statement

Traffic sign recognition is a safety-critical component of modern driver assistance systems and autonomous vehicles. A deployed classifier must correctly identify signs under varying illumination, partial occlusion, motion blur, and a wide range of distances — all while distinguishing between 43 visually distinct classes with very low tolerance for error. A misclassified speed limit or stop sign could have direct consequences for road safety.

The German Traffic Sign Recognition Benchmark (GTSRB) is the standard dataset for this task. It presents several concrete challenges: the 39,209 training images span wildly different resolutions (15×15 to 250×250 pixels), the class distribution is highly skewed (11× imbalance between rarest and most frequent), and some classes share nearly identical visual structures that are only distinguishable by small symbolic details.

### 1.2 Our Approach

We decided to build a series of convolutional neural networks from scratch rather than defaulting immediately to a large pretrained model. Our reasoning was that GTSRB is a constrained, single-domain problem — images are pre-cropped, classes are visually structured, and the dataset is large enough relative to the task complexity. We wanted to understand how much a compact, purpose-built CNN can achieve before resorting to the computational overhead of transfer learning.

The project is structured in six stages: dataset analysis, preprocessing, a baseline model, architectural variants, evaluation, and this report. For the architectural variants, we deliberately chose one change at a time — depth, activation function, downsampling method, and transfer learning — so that we could isolate the effect of each individual design decision.

### 1.3 Expectations

Before training we expected the baseline to perform well — the combination of pre-cropped images, clear sign structure, and augmentation seemed favorable. What we did not know upfront was how much each individual architectural change would matter. Our hypothesis was that depth (more convolutional blocks) would provide the most benefit, since GTSRB requires recognizing fine symbolic details that hierarchical features should capture well. We expected transfer learning to be useful primarily for the rarest classes but not to dominate overall. We expected activation function and downsampling choices to have smaller effects, since BatchNorm was already stabilizing training. These expectations were largely confirmed, as discussed in Section 7.

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

The sample grid illustrates the visual diversity within the dataset. Even within a single class, images vary in brightness, contrast, viewing angle, and background — motivating careful preprocessing and augmentation.

### 2.4 Data Source

The GTSRB dataset was introduced by Stallkamp et al. in their 2012 paper *"Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition"* (Neural Networks, 32:323–332). The benchmark was presented at the IJCNN 2011 competition, where the best entry achieved 99.46% — surpassing human-level performance of 98.84%. GTSRB is therefore a well-established benchmark and near-perfect CNN performance is consistent with the literature.

---

## 3. Data Preprocessing

### 3.1 Data Split

The 39,209 training images are divided into three non-overlapping subsets using a fixed random seed (42) for reproducibility:

| Split | Fraction | Images |
|-------|----------|--------|
| Training | 70% | 27,447 |
| Validation | 15% | 5,881 |
| Test | 15% | 5,881 |

![Per-class sample distribution across training, validation, and test splits](results/preprocessing_split_distribution.png)

Note: GTSRB provides an official test set, but since its ground-truth labels were not used in this project pipeline, the original 39,209 labelled training images were stratified into train, validation, and held-out test subsets. Reported accuracy figures therefore reflect performance on this internal split and should not be directly compared with results evaluated on the official benchmark test set.

The validation set is used during training to monitor generalization and apply early stopping. The test set is held out entirely and evaluated only once per model, ensuring reported results are not inflated by repeated evaluation.

### 3.2 Image Transformations

All images are resized to **32×32 pixels** before processing. We chose this resolution as a deliberate tradeoff: it is compact enough for fast training on consumer hardware while retaining enough spatial detail for the model to distinguish sign shapes, symbols, and colors. Higher resolutions like 64×64 would increase computational cost substantially without guaranteeing accuracy gains — a tradeoff we revisit in the Future Work section.

**Training transforms** apply stochastic augmentations to increase effective diversity:

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| Random Rotation | ±15° | Simulates tilted camera angles |
| Color Jitter | brightness ±0.4, contrast ±0.4, saturation ±0.3 | Simulates lighting and weather variation |
| Random Affine | translate ±10% | Simulates off-center sign placement |
| Normalize | mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629) | Centers input distribution |

**Validation and test transforms** are fully deterministic — only resize, convert to tensor, and normalize. No augmentation is applied during evaluation so that measured accuracy honestly reflects model performance on unmodified inputs.

### 3.3 Normalization

Pixel values are converted from [0, 255] to floating-point [0.0, 1.0], then normalized per channel using mean and standard deviation computed from the GTSRB training set. Without normalization, large differences in pixel scales across channels distort the loss surface and slow convergence.

### 3.4 Data Augmentation as Regularization

Augmentation artificially increases the effective diversity of the training set. This is particularly important for the rarest sign categories with fewer than 200 training samples, where the model would otherwise see the same images repeatedly and memorize them.

### 3.5 Mini-Batch Loading and Early Stopping

Images are fed to the model in mini-batches of size 64. Mini-batch training introduces stochasticity into the optimization, which helps the optimizer escape poor local minima. Early stopping with patience 5 halts training when validation accuracy does not improve for five consecutive epochs, restoring the best-seen checkpoint for evaluation.

---

## 4. Baseline Model

### 4.1 Architecture

The baseline CNN consists of three convolutional blocks followed by a fully connected classifier. The architecture diagram below (left) shows the full layer sequence with feature map dimensions at each stage.

![Architecture comparison: Baseline CNN (left) vs. Deep CNN (right)](results/diagrams/architecture_comparison.png)

**Total trainable parameters: 629,291**

Each convolutional block applies a 3×3 convolution with padding=1 (preserving spatial dimensions), followed by Batch Normalization, ReLU activation, and 2×2 MaxPooling that halves the spatial dimensions. The classifier uses Dropout(0.5) to regularize and outputs raw logits — no softmax is applied because CrossEntropyLoss handles the log-softmax internally, which is numerically more stable.

### 4.2 How Data Flows Through the Network

To make the architecture concrete, here is how a single 32×32 RGB image passes through the baseline CNN step by step:

**Input:** 3 × 32 × 32 (3 color channels, 32×32 pixels)

**Block 1** — Conv(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2): `32 × 32 × 32` → `32 × 16 × 16`

**Block 2** — Conv(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2): `64 × 16 × 16` → `64 × 8 × 8`

**Block 3** — Conv(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2): `128 × 8 × 8` → `128 × 4 × 4`

**Flatten:** `128 × 4 × 4` → `2,048`-dimensional feature vector

**FC1** — Linear(2048→256) → ReLU → Dropout(0.5)

**FC2** — Linear(256→43) → logits for 43 classes

Each MaxPool step halves the spatial resolution while the convolutions double the number of feature maps, so the network progressively trades spatial detail for richer feature representation. By the time the feature map reaches the classifier, each of the 2,048 values encodes a learned local pattern from the original image.

### 4.3 Parameter Selection Rationale

We did not choose the architecture parameters arbitrarily. The 3×3 convolution kernel is the standard choice in modern CNNs — it captures local spatial patterns with minimal parameters (9 weights vs. 25 for 5×5) and stacking multiple such layers achieves the same receptive field as a single large kernel at lower cost. Doubling filter counts per block (32→64→128) follows the established convention that deeper layers should represent more complex, higher-dimensional feature spaces.

For the training configuration: Adam was preferred over plain SGD because it adapts learning rates per parameter, which typically leads to faster convergence on image classification tasks. The initial learning rate of 1×10⁻³ is the widely used default for Adam. Batch size 64 balances GPU memory usage, gradient noise, and training speed. ReduceLROnPlateau with patience 3 was added because we observed validation loss plateauing mid-training in early experiments — the scheduler reliably resumed progress by halving the rate.

### 4.4 Results

Two runs were conducted with different random seeds to verify stability:

| Seed | Best Val Accuracy | Test Accuracy | Test Loss |
|------|------------------|--------------|-----------|
| 42   | 98.78%           | 98.55%       | 0.0621    |
| 123  | 99.15%           | 99.29%       | 0.0451    |

![Baseline training curves (seed 42): training and validation loss and accuracy over epochs](results/task04/baseline_loss_curve_seed-42.png)

The loss curves show smooth convergence with no signs of severe overfitting — the train/val gap remains small throughout.

### 4.5 Why High Baseline Accuracy is Expected

The near-perfect baseline accuracy reflects properties intrinsic to GTSRB rather than overfitting or data leakage. Traffic signs are designed to be maximally distinguishable — the dataset exhibits high inter-class variability (each class looks structurally different) with low intra-class variability (all instances share the same shape, color, and symbol). Additionally, GTSRB images are pre-cropped to the sign bounding box, so the model solves pure classification rather than the harder joint detection-and-classification task. Stallkamp et al. (2012) report a human recognition rate of **98.84%** — below our baseline — confirming that near-perfect performance is consistent with established results.

---

## 5. Model Improvements

### 5.1 Overview and Expectations

After establishing the baseline, we designed four architectural variants. Rather than testing arbitrary changes, each variant isolates exactly one design decision so that performance differences can be attributed clearly. Our expectations going into each variant:

**Deep CNN** — We expected this to be the most impactful variant. A fourth convolutional block (128→256 filters) gives the network capacity to learn more abstract features, which should help with the fine symbolic differences between visually similar classes. We predicted an accuracy gain of around 0.2–0.5 pp.

**LeakyReLU CNN** — We expected a small but positive effect. Dead neurons are a known issue with ReLU, but BatchNorm partially mitigates this by keeping activations in a healthy range. We were not confident this would make a measurable difference.

**Stride CNN** — We expected marginal change. Replacing MaxPool with learned strided convolutions is theoretically more flexible, but MaxPool already performs well on structured inputs like traffic signs. We expected similar or slightly lower accuracy with faster training.

**MobileNetV2** — We expected this to be competitive overall but not necessarily the best. ImageNet features are useful for general vision, but GTSRB is a specialized domain with its own color and shape vocabulary. We predicted it would excel on rare classes but lose out on parameter efficiency.

All variants were trained under identical conditions: same optimizer (Adam, lr=1×10⁻³), scheduler, augmentation pipeline, data split, and up to 20 epochs with early stopping.

| Model | Test Accuracy | Wrong / 5881 | Parameters | Training Time |
|-------|-------------|:---:|-----------|:---:|
| Baseline CNN | 99.49% | 30 | 629,291 | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936,235** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2,562,859 | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629,291 | 271.5 s |
| Stride CNN | 99.52% | 28 | 823,051 | 236.9 s |

Because the test set contains 5,881 samples, a difference of 0.1 percentage points corresponds to approximately six images. Differences below this threshold should not be overinterpreted as meaningful improvements.

![Model comparison summary: test accuracy, accuracy vs. parameters, and training time](results/task05/model_comparison_summary.png)

![Training accuracy curves for all five model variants](results/task05/model_comparison_curves.png)

### 5.2 Variant A — Deep CNN

The Deep CNN adds a fourth convolutional block (128→256 filters, 3×3 kernels, BatchNorm, ReLU, MaxPool) and expands the classifier from 256 to 512 hidden units. All other settings — Dropout(0.5), same augmentation pipeline and data split — are identical to the baseline. This variant achieves the **highest test accuracy of 99.81%** — only 11 wrong predictions out of 5,881 — with only a 49% parameter increase and nearly identical training time (284 s vs. 276 s).

### 5.3 Variant B — MobileNetV2 (Transfer Learning)

MobileNetV2 (Sandler et al., 2018), pretrained on ImageNet (1.2 million images, 1,000 classes), was used with a custom two-layer classifier head adapted for the 43 GTSRB classes. Inputs were resized to 32×32 pixels and normalized using GTSRB channel statistics; all backbone weights were fine-tuned during training. MobileNetV2 achieves 99.66% but requires **4× more parameters** and nearly **twice the training time** for only a 0.17 pp gain over baseline — on this dataset, transfer learning does not justify the added cost.

### 5.4 Variant C — LeakyReLU CNN

This variant replaces all ReLU activations with Leaky ReLU (slope = 0.01). Despite the theoretical advantage of preventing dead neurons, it achieves 99.46% — marginally below the baseline (99.49%). With BatchNorm stabilizing activations throughout, dead neurons are not a significant bottleneck at this scale.

### 5.5 Variant D — Stride CNN

Instead of fixed MaxPool, the Stride CNN uses strided convolutions (stride=2) for downsampling — the network learns how to subsample rather than applying a fixed maximum rule. It achieves 99.52% and is the **fastest to train** (236.9 s), making it attractive under compute constraints. The accuracy difference is within the noise threshold.

### 5.6 Parameter Sensitivity

To understand how sensitive the results were to key hyperparameters, we ran Bayesian hyperparameter search using Optuna (Akiba et al., 2019) with a Tree-structured Parzen Estimator across learning rate (1×10⁻⁴ to 1×10⁻²), dropout (0.2–0.6), batch size (32/64/128), optimizer (Adam/SGD), and weight decay (1×10⁻⁵ to 1×10⁻³).

The search consistently identified the Adam optimizer with a learning rate in the range 5×10⁻⁴ to 2×10⁻³ as the most stable configuration. Trials using SGD converged more slowly and were more sensitive to the learning rate. Batch size had surprisingly little effect on final accuracy — both 32 and 128 reached comparable results, though 128 was faster per epoch. Dropout in the range 0.3–0.5 was consistently preferred; values below 0.3 led to slightly higher validation loss from overfitting. This exploratory search confirmed that our manually chosen defaults (Adam, lr=1×10⁻³, batch=64, dropout=0.5) sit in a well-performing region of the hyperparameter space.

### 5.7 Latent Space Visualisation

To understand what the network learned internally, feature vectors were extracted from the penultimate layer of the baseline CNN and projected to two dimensions using t-SNE (van der Maaten & Hinton, 2008) with perplexity 30. If the 43 classes form distinct clusters in the 2D projection, the network has learned to separate them in its internal representation — providing interpretable evidence beyond accuracy numbers alone. Visually similar classes such as different speed limits would be expected to appear close together; structurally distinct classes like Stop and Yield should be well separated.

### 5.8 Autoencoder for Anomaly Detection

A key limitation of any classifier is that it always assigns an input to one of its known classes — even when the input is outside the training distribution. We implemented a convolutional autoencoder as a complementary anomaly detection mechanism, applying the concept from Lecture 7.

The encoder compresses 3×32×32 images through three convolutional blocks down to a 128-dimensional latent vector; a mirrored decoder reconstructs the image. Training is unsupervised — the objective is to minimise the per-pixel MSE between input and reconstruction:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{a}_i - a_i)^2$$

After training, reconstruction error serves as an anomaly score. A threshold at the 95th percentile of the validation error distribution flags inputs as anomalous. This component was implemented as a proof-of-concept for Lecture 7; quantitative evaluation on out-of-distribution samples was beyond the scope of this project.

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

### 6.2 Confusion Matrix

![Normalized confusion matrix of the Deep CNN on the test set](results/task06/deep/confusion_matrix_normalized.png)

The confusion matrix is strongly diagonal. The few off-diagonal entries are concentrated among visually similar sign pairs — different speed limit signs and warning signs with comparable layouts.

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

These classes share a common characteristic: they are visually similar to neighbouring classes, with subtle differences that are difficult to resolve at 32×32 resolution.

### 6.4 Precision and Recall

![Precision and recall per class for the Deep CNN](results/task06/deep/precision_recall_per_class.png)

Precision and recall are consistently high across all 43 classes. The few reduced scores correspond to the visually ambiguous categories identified above.

### 6.5 Misclassified Examples

![High-confidence misclassifications: cases where the model was wrong but confident](results/task06/deep/misclassifications_top_confidence.png)

The 11 misclassified test images are concentrated in genuinely hard cases — degraded quality, partial occlusion, or strong visual similarity to another class. There are no systematic failures of an entire category.

### 6.6 Bias Analysis

A critical concern for deployment is whether the model performs disproportionately worse on underrepresented classes. We split the 43 classes into the 10 most and 10 least frequent by training count.

![Mean test accuracy for frequent vs. rare traffic sign classes](results/task06/deep/bias_analysis_mean_accuracy.png)

| Group | Training images (avg.) | Mean Test Accuracy |
|-------|----------------------|-------------------|
| Frequent classes (top 10) | ~1,374 per class | 99.87% |
| Rare classes (bottom 10) | ~169 per class | 99.52% |
| Gap | — | **0.34 percentage points** |

The 0.34 pp gap suggests augmentation and training mitigate class imbalance effectively — notably, several of the rarest classes achieve 100% test accuracy. These figures are based on a single run and split; the absolute gap may vary. A model accurate on average but failing on rare classes would be unsuitable for deployment — rare signs require reliable recognition precisely because they appear infrequently.

### 6.7 Robustness Testing

| Condition | Test Accuracy | Δ vs. Clean |
|-----------|-------------|:-----------:|
| Clean | 99.81% | — |
| Gaussian Blur (kernel=5) | 97.01% | −2.80 pp |
| Gaussian Noise (σ=0.1) | 71.86% | **−27.95 pp** |

The model handles blur well but suffers a dramatic drop under Gaussian noise. The σ=0.1 noise level is applied to normalized pixel values in [0,1], representing moderate corruption. This is the most significant limitation for real-world deployment with low-quality sensors.

### 6.8 Grad-CAM Interpretability

![Grad-CAM visualizations: image regions that most influenced the model's predictions](results/task06/deep/gradcam_examples.png)

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) highlights the image regions that most strongly influenced predictions. The visualizations provide evidence that the model attends to the relevant sign regions — shape, symbol, and color — rather than background artifacts, suggesting predictions are based on the sign content itself.

---

## 7. Discussion

### 7.1 Summary of Findings

All five models exceed 99% test accuracy, providing evidence that CNN-based classifiers are well-suited to GTSRB. The results largely matched our initial expectations:

**Depth helps, as expected.** The Deep CNN gains 0.32 pp at minimal additional cost — the most cost-effective improvement and in line with our prediction.

**Transfer learning is not necessary here.** MobileNetV2 gains only 0.17 pp at 4× the parameters and 2× the training time, confirming our expectation that the GTSRB dataset is large enough for from-scratch learning.

**Activation and downsampling choices have minimal impact**, as we anticipated — BatchNorm dominates stabilization.

**Class imbalance is handled effectively.** The 0.34 pp accuracy gap between frequent and rare classes provides evidence that augmentation mitigates imbalance without explicit reweighting.

**Noise robustness is the main open challenge.** The 27.95 pp drop under Gaussian noise was the one result that exceeded our expected severity.

### 7.2 Assumptions, Limitations, and Biases

**Fixed 32×32 resolution.** Some visually similar classes might be more reliably distinguished at higher resolution, at the cost of larger models and longer training.

**Single run per improved model.** The improved variants were each trained once. Performance estimates would be more reliable with multiple independent runs.

**Clean training data.** No noise or blur augmentation was applied during training, which directly explains the poor noise robustness (Section 6.7).

**Fixed data split.** The 70/15/15 split is applied once. Cross-validation would provide more robust estimates but was not applied due to the cost of training five variants.

**Selection bias.** GTSRB was recorded exclusively on German roads. Signs from other countries, extreme conditions (heavy rain, snow, night), or differently styled variants are absent from the training distribution.

**Class frequency bias.** The 11× imbalance creates a systematic risk of disproportionate optimisation for common classes. Although the measured gap is small, failures on rare signs are safety-critical precisely because they appear infrequently.

**Representation bias.** The dataset contains no damaged, faded, or vandalized signs. Gaussian noise tests partially probe this gap but do not cover realistic damage such as occlusion or physical deformation.

**Measurement bias.** All images were captured from a single camera system. Performance may degrade on data from different sensors with different optics or noise profiles.

### 7.3 Suitability Assessment

For the purpose of this course project, the approach is fully suitable. The Deep CNN achieves near-perfect accuracy (99.81%), generalizes well across class frequencies, and Grad-CAM provides evidence that predictions are based on sign-relevant features. For real-world deployment, the noise sensitivity would need to be addressed — augmenting training with noise and blur perturbations is the most practical first step.

---

## 8. Future Work

The current system classifies pre-cropped traffic sign images. Several extensions would move it toward real-world applicability.

**Noise and blur augmentation** would directly close the 27.95 pp robustness gap identified in Section 6.7 at minimal additional cost.

**Object detection integration** is the most impactful step toward deployment. The current pipeline assumes pre-cropped inputs, which does not hold in real driving footage. A combined system would first detect sign locations in the full frame, then classify each crop. The extended system architecture would look as follows:

```
┌─────────────────────────────────────────────────┐
│                Full Pipeline                     │
│                                                  │
│  CameraFrame → [Detector] → BoundingBoxes        │
│                                  │               │
│                             [Classifier]         │
│                                  │               │
│                          ClassLabel + Confidence │
│                                  │               │
│                          [AnomalyFilter]         │
│                                  │               │
│                        Accept / Reject           │
└─────────────────────────────────────────────────┘
```

The Detector (e.g. YOLO) would be a separate trained module; the Classifier and AnomalyFilter (autoencoder) already exist in this project. Connecting them is the main engineering gap.

**Higher input resolution** (64×64) would preserve more spatial detail and likely reduce confusion between visually similar classes such as Pedestrians and Bicycles crossing.

**Cross-validation** over multiple splits would provide statistically more reliable performance estimates, particularly for rare classes.

**Domain adaptation** — fine-tuning on signs from other countries or adverse weather conditions — would reduce selection bias and improve generalisability beyond German roads.

---

## 9. Conclusion

The central question was whether a compact from-scratch CNN can solve GTSRB reliably. It can: the Deep CNN reaches **99.81% test accuracy** with 936K parameters, misclassifying only 11 of 5,881 test images. Our structured comparison confirms that depth is the most cost-effective improvement, while transfer learning, alternative activations, and learned downsampling offer no meaningful advantage on a dataset of this size and structure.

The pipeline handles class imbalance well (0.34 pp accuracy gap) and Grad-CAM provides evidence that the model learns sign-relevant features. The principal limitation is noise sensitivity (−27.95 pp at σ=0.1), which is the clearest remaining gap between benchmark performance and real-world reliability. Noise augmentation during training and integration with an object detector are the two most impactful next steps toward deployment.

---

## References

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32, 323–332. https://doi.org/10.1016/j.neunet.2012.02.016

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.

---

## Appendix: Generated Artifacts

| Task | Key Output Files |
|------|-----------------|
| Task 02 | `results/class_mapping.csv`, `results/task03/class_distribution.png` |
| Task 03 | `results/preprocessing_stats.json`, `results/preprocessing_sample_grid.png` |
| Task 04 | `models/baseline.pth`, `results/task04/baseline_history_seed-42.json`, `results/task04/baseline_loss_curve_seed-42.png` |
| Task 05 | `models/deep_cnn.pth`, `results/task05/model_comparison.json`, `results/task05/model_comparison_summary.png` |
| Task 06 | `results/task06/deep/evaluation_summary.json`, `results/task06/deep/gradcam_examples.png`, `results/task06/deep/confusion_matrix_normalized.png`, `results/task06/deep/bias_analysis_mean_accuracy.png` |
