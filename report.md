# CNN Traffic Sign Classification: Final Report
**German Traffic Sign Recognition Benchmark (GTSRB)**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data and Preprocessing](#2-data-and-preprocessing)
3. [Baseline Model](#3-baseline-model)
4. [Model Improvements](#4-model-improvements)
5. [Results and Discussion](#5-results-and-discussion)
6. [Conclusion and Future Work](#6-conclusion-and-future-work)

---

## 1. Introduction 

### 1.1 Problem Context and Motivation

Traffic sign recognition is an important component of modern driver assistance systems and autonomous driving pipelines. In real-world applications, a classifier must recognize signs under changing illumination, partial occlusion, motion blur, and different viewing distances. Because traffic signs communicate legally and safety-relevant instructions, misclassifying a speed limit, priority sign, or stop sign could lead to incorrect driving decisions, making reliable classification both a technical and a safety-relevant challenge.

To study this problem in a controlled and reproducible way, this project uses the German Traffic Sign Recognition Benchmark (GTSRB), a widely used dataset for traffic sign classification. A detailed description of the dataset and preprocessing pipeline is provided in Section 2.

### 1.2 Project Approach

This project investigates how far compact convolutional neural networks can be pushed on the GTSRB classification task before comparing them with a pretrained model. We trained CNNs from scratch first because the dataset structure supports it: GTSRB images are already cropped to the sign bounding box, the visual domain is narrow, and the classes follow standardized color and shape patterns. Under these conditions, a purpose-built CNN may achieve strong performance with lower computational cost than transfer learning.

The project compares five model variants by changing one design decision at a time, covering network depth, activation function, downsampling strategy, and transfer learning. This controlled setup makes it possible to attribute performance differences to specific architectural choices.

### 1.3 Goal of the Project

The goal of this project is to systematically evaluate how different CNN architectures perform on the GTSRB traffic sign classification task and to identify which design decisions drive performance improvements. This includes assessing generalization across all 43 sign classes, bias toward frequent versus rare classes, and robustness under simulated image degradations such as noise and blur. Beyond raw accuracy, the evaluation uses Grad-CAM visualizations and latent space analysis to provide interpretable evidence for model behavior.

### 1.4 Assumptions

This project operates under several scope-defining assumptions. All GTSRB images are pre-cropped to the sign bounding box, so the models perform pure image classification and are never required to locate a sign within a larger scene. Because the official GTSRB test labels were not available in our pipeline, all accuracy figures are computed on an internal 15% hold-out split and are not directly comparable to the official competition leaderboard. The comparison between our five model variants remains valid because all models are evaluated under identical conditions on the same internal split. Each image is assumed to contain exactly one sign belonging to one of the 43 mutually exclusive GTSRB classes, and the provided labels are assumed to be correct. Finally, the GTSRB training data is treated as representative of the classification task, though this assumption is limited: the dataset was recorded exclusively on German roads from a single camera system under normal weather conditions, which constrains generalizability to other real-world scenarios.

---

## 2. Data and Preprocessing 

### 2.1 Dataset Overview

The GTSRB dataset (Stallkamp et al., 2012) was recorded from a car-mounted camera on German roads and contains **39,209 training images** across **43 traffic sign classes** (IDs 0–42). Image dimensions vary widely, from 25×25 to 243×225 pixels, with a mean of approximately 50×50 pixels. The dataset covers speed limit signs, prohibitory signs, mandatory direction signs, warning signs, and right-of-way signs. Many classes share the same basic shape and differ only in a small internal detail, such as the numeral on a speed limit sign or the icon inside a warning triangle, making inter-class similarity one of the primary classification challenges.

### 2.2 Class Distribution and Visual Challenges

![Class distribution across all 43 GTSRB traffic sign categories](results/task03/class_distribution.png)

*The x-axis shows class IDs 0–42. The full class-ID-to-name mapping is provided in Appendix B.*

The dataset is not uniformly distributed. The most frequent class is Speed limit (50 km/h) (class ID 2) with **2,250 images**, while Speed limit (20 km/h) (class ID 0), Dangerous curve left (ID 19), and Go straight or left (ID 37) are the rarest, each with **210 images**, giving an imbalance ratio of approximately **10.7×**. After the 70% training split, the rarest classes contribute roughly 147 training images each. This imbalance risks weaker generalization on underrepresented classes and motivates the per-class accuracy analysis in Section 5.

![One representative image per class](results/task03/sample_images_by_class.png)

*IDs 0–42, left to right.*

The sample grid illustrates two complementary challenges. Within each class, images vary considerably in brightness, contrast, and viewing angle (intra-class variation the model must learn to ignore). Across classes, many signs share the same shape and differ only in one small detail (inter-class similarity that is the primary source of potential misclassification).

### 2.3 Benchmark Context and Evaluation Setup

GTSRB was used in the IJCNN 2011 competition, where participants were evaluated on a separate official test set of **12,630 images**. The best submitted system, a committee of CNNs, reached **99.46% accuracy**, surpassing the reported human recognition rate of **98.84%** (Stallkamp et al., 2012). Because the official test labels were not available in our pipeline, all accuracy figures in this report are computed on an internal 15% hold-out split and are not directly comparable to the competition leaderboard. They are valid for comparing our five model variants because all are evaluated under identical conditions on the same internal split.

### 2.4 Data Split

The 39,209 labelled images are divided into three non-overlapping subsets:

| Split | Fraction | Images |
|-------|----------|--------|
| Training | 70% | 27,447 |
| Validation | 15% | 5,881 |
| Test | 15% | 5,881 |

The training set is used to update model weights; the validation set monitors generalization during training; the test set is reserved for a single final evaluation. The split is performed with `torch.utils.data.random_split` and a fixed seed of 42, making it reproducible but not stratified. Exact class proportions across subsets are not guaranteed, but at this dataset size they are expected to remain close to the original distribution.

![Per-class sample distribution across training, validation, and test splits](results/preprocessing_split_distribution.png)

*The x-axis shows class IDs 0–42. Each bar group shows the image count per class per split.*

### 2.5 Preprocessing and Augmentation

All images are resized to **32×32 pixels** to provide a uniform input size. For the training set, random augmentations are applied on each pass; for validation and test, transforms are fully deterministic.

| Transform | Applied to | Parameters |
|-----------|------------|------------|
| Resize | All splits | 32×32 px |
| Random Rotation | Train only | ±15° |
| Color Jitter | Train only | brightness=0.4, contrast=0.4, saturation=0.3 |
| Random Affine | Train only | translate=±10% |
| ToTensor | All splits | Converts PIL image to float tensor |
| Normalize | All splits | mean=(0.3337, 0.3064, 0.3171), std=(0.2672, 0.2564, 0.2629) |

Each augmentation simulates a real-world variation: rotation accounts for tilted camera angles, color jitter for changing lighting and weather, and affine translation for off-center framing. Because augmentations are sampled independently on each epoch, the model is unlikely to encounter the exact same pixel pattern twice, which acts as a regularization mechanism and is particularly beneficial for the rarest classes with only approximately 147 training images each. Normalization subtracts the per-channel mean and divides by the per-channel standard deviation, centering inputs near zero for stable gradient flow. Both statistics were computed from the GTSRB training data and applied identically to all splits. Validation and test images receive only the deterministic transforms (resize, ToTensor, normalize), ensuring that evaluation reflects performance on images that have not been randomly distorted.

---

## 3. Baseline Model

### 3.1 Architecture and Design Decisions

Traffic signs are spatial visual objects, making a Convolutional Neural Network the appropriate model class. CNNs learn spatially local patterns through small filters and build up increasingly abstract representations layer by layer, making them far more efficient and better suited to image classification than fully connected networks.

The baseline is a compact three-block CNN with **629,291 trainable parameters**.

![Architecture comparison: Baseline CNN (left) vs. Deep CNN (right)](results/diagrams/architecture_comparison.png)

Three convolutional blocks are a natural depth for 32×32 inputs: after three 2×2 MaxPool operations the spatial dimensions reduce to 4×4, providing enough compression to capture global structure while retaining sufficient detail for classification. Each block follows the pattern: 3×3 convolution with padding 1 (to preserve spatial dimensions before pooling), Batch Normalization to stabilize activations, ReLU, and 2×2 MaxPooling to halve spatial resolution. The filter count increases from 32 in Block 1 to 64 in Block 2 and 128 in Block 3: later layers combine more primitive features into increasingly abstract representations and therefore require more channels as spatial dimensions shrink.

After the three blocks, the feature maps are flattened to a 2,048-dimensional vector and passed through a fully connected classifier: Linear(2,048 to 256), ReLU, Dropout(0.5), and Linear(256 to 43). Dropout regularizes the classifier by randomly zeroing half its activations during training. The final layer produces 43 raw logits, one per traffic sign class. CrossEntropyLoss handles the softmax conversion internally during training.

### 3.2 Forward Pass: From Image to Prediction

A single 32×32 RGB image enters as a 3×32×32 tensor and passes through the following stages:

| Stage | Output Shape | Representation |
|-------|-------------|----------------|
| Input | 3 × 32 × 32 | Raw pixel values |
| After Block 1 | 32 × 16 × 16 | Edges, color gradients, simple textures |
| After Block 2 | 64 × 8 × 8 | Corners, curves, color regions |
| After Block 3 | 128 × 4 × 4 | Shapes, symbols, structural patterns |
| After Flatten | 2,048 | Full feature summary |
| After FC1 | 256 | Compressed, class-discriminative representation |
| After FC2 | 43 | One confidence score per traffic sign class |

Spatial resolution decreases while channel count increases at every stage. By the time the representation reaches the classifier, the 2,048-dimensional vector encodes a compact summary of the features the network learned to use for class separation.

### 3.3 Training Configuration

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| Optimizer | Adam | Adapts learning rate per parameter using past gradients and their variance; more stable and less sensitive to initial learning rate than plain SGD |
| Initial learning rate | 1×10⁻³ | Standard Adam default; confirmed effective by the hyperparameter search in Section 4.6 |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) | Halves the learning rate when validation loss does not improve for 3 consecutive epochs, allowing finer updates near convergence |
| Loss function | CrossEntropyLoss | Standard multi-class classification loss; handles softmax internally |
| Batch size | 64 | Balances gradient stability, memory usage, and training speed |
| Max epochs | 30 | Upper bound; early stopping typically engages before this |
| Early stopping patience | 5 | Stops training after 5 epochs without validation accuracy improvement |

Images are processed in mini-batches of 64, with model weights updated after each batch. Stochastic mini-batch updates introduce controlled noise into the optimization that helps avoid poor local minima. Early stopping monitors validation accuracy and halts training if it does not improve for five consecutive epochs, restoring the best-performing checkpoint for final evaluation.

### 3.4 Baseline Results and Interpretation

**Why high accuracy is expected on GTSRB.** Traffic signs are designed for fast reliable recognition using standardized shapes, bold colors, and unambiguous symbols. All images are pre-cropped to the sign bounding box, so the model performs pure classification of well-framed images rather than detection. Under these conditions a compact CNN is expected to achieve strong performance.

**Two-seed stability runs.** The baseline was trained twice with different random seeds to verify stability. Both runs used a maximum of 10 epochs:

| Seed | Best Val Accuracy | Test Accuracy | Test Loss |
|------|------------------|--------------|-----------|
| 42   | 98.78%           | 98.55%       | 0.0621    |
| 123  | 99.15%           | 99.29%       | 0.0451    |

Both seeds produce comparable accuracy, confirming that results are not dependent on a particular random initialization or mini-batch ordering. The training loss was still decreasing at epoch 10, indicating that these models had not yet fully converged; the 10-epoch cap was a practical constraint for this exploratory run.

![Baseline training curves (seed 42): training and validation loss and accuracy over epochs](results/task04/baseline_loss_curve_seed-42.png)

The seed-42 training curves show training and validation loss decreasing steadily, with both curves tracking closely throughout. There is no divergence between training and validation loss that would indicate overfitting.

**Canonical baseline for model comparison.** The model comparison in Section 4 uses a separate run in which all five model variants were trained under identical conditions with a maximum of 20 epochs. In that run the baseline reached **99.49% test accuracy** (30 wrong out of 5,881 test images). This is the canonical figure for all model comparisons. The two seed runs above serve only to confirm stability.

Stallkamp et al. (2012) report an average human recognition rate of 98.84% on the official GTSRB benchmark. Baseline accuracies around 99% on our internal split are therefore not surprising, although the figures are not directly comparable because they are based on different evaluation sets.

---

## 4. Model Improvements

### 4.1 Overview and Expectations

After establishing the baseline, we designed four architectural variants. Rather than testing arbitrary changes, each variant isolates exactly one design decision so that performance differences can be attributed clearly. Before training, our expectations were:

**Deep CNN:** most likely to improve. A fourth convolutional block gives the network capacity to learn more abstract features, helpful for fine symbolic differences between similar classes. We predicted a gain of roughly 0.2–0.5 pp.

**LeakyReLU CNN:** small effect expected. Dead neurons are a known ReLU issue, but BatchNorm already keeps activations healthy. We were not confident this would make a measurable difference.

**Stride CNN:** marginal change expected. Learned downsampling is theoretically more flexible than MaxPool, but MaxPool already works well on structured inputs. We expected similar accuracy with faster training.

**MobileNetV2:** competitive but not dominant. ImageNet features generalize broadly but GTSRB is a specialized domain. We expected it to excel on rare classes but lose out on parameter efficiency.

All variants were trained under identical conditions: Adam (lr=1×10⁻³), same scheduler, same augmentation, same data split, up to 20 epochs with early stopping.

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|-------|-------------|:---:|-----------|:---:|
| Baseline CNN | 99.49% | 30 | 629,291 | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936,235** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2,562,859 | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629,291 | 271.5 s |
| Stride CNN | 99.52% | 28 | 823,051 | 236.9 s |

Note: because the test set contains 5,881 samples, a difference of 0.1 percentage points corresponds to approximately six images. Differences below this threshold should not be overinterpreted as meaningful improvements.

![Model comparison summary: test accuracy, accuracy vs. parameters, and training time](results/task05/model_comparison_summary.png)

![Training accuracy curves for all five model variants](results/task05/model_comparison_curves.png)

### 4.2 Variant A: Deep CNN

**What changed:** A fourth convolutional block was added (128→256 filters, 3×3 kernels, BatchNorm, ReLU, MaxPool), and the fully connected classifier was expanded from 256 to 512 hidden units. All other settings remain identical to the baseline.

**Why we expected this to help:** The baseline stops after three convolutional blocks, which gives the network a receptive field and feature hierarchy deep enough for simple patterns but potentially insufficient for the finer symbolic distinctions between sign types. A fourth block compresses the spatial resolution to 2×2 and pushes the number of feature channels to 256, forcing the network to learn more abstract, class-discriminative representations in the penultimate layer.

**Result:** The Deep CNN achieves **99.81% test accuracy**, with only 11 wrong predictions out of 5,881. This confirms our prediction. With only a 49% increase in parameters and nearly identical training time (284 s vs. 276 s), it is the most cost-effective improvement we found.

### 4.3 Variant B: MobileNetV2 (Transfer Learning)

**What changed:** Instead of a custom CNN trained from scratch, we used MobileNetV2 (Sandler et al., 2018) pretrained on ImageNet, a general-purpose image dataset with 1.2 million images and 1,000 classes. A custom two-layer classifier head was attached and all weights including the backbone were fine-tuned on GTSRB. Inputs were resized to 32×32 and normalized using GTSRB channel statistics.

**Why we chose this:** Transfer learning is motivated by the insight that low-level visual features such as edges, textures, and color gradients are shared across many image domains. The pretrained backbone provides a strong starting point, especially for the rarest GTSRB classes with fewer than 200 training images where learning from scratch may not converge well.

**Result:** MobileNetV2 achieves 99.66%, which is better than the baseline but at 4× the parameters (2.56M vs. 629K) and nearly 2× the training time (519 s vs. 276 s). For only a 0.17 pp gain, the additional cost is not justified on this dataset. The GTSRB training set is large enough for compact CNNs to learn excellent representations without ImageNet pretraining.

### 4.4 Variant C: LeakyReLU CNN

**What changed:** All ReLU activations were replaced with Leaky ReLU (negative slope = 0.01). Everything else is identical to the baseline.

**Why we considered this:** Standard ReLU outputs zero for any negative input, meaning its gradient is also zero. If a neuron's inputs are consistently negative, which can happen due to unlucky weight initialization or aggressive weight updates, it permanently stops learning. This is the "dead neuron" problem. Leaky ReLU prevents it by allowing a small gradient (0.01 × input) for negative values, keeping all neurons active.

**Result:** 99.46%, marginally below the baseline (99.49%). With BatchNorm normalizing activations before each ReLU, inputs are kept in a healthy range and dead neurons are not a significant problem at this scale. The theoretical advantage of Leaky ReLU does not materialize here.

### 4.5 Variant D: Stride CNN

**What changed:** MaxPool layers were replaced with strided convolutions (stride=2) for spatial downsampling. Instead of applying a fixed maximum rule, the network learns its own downsampling weights.

**Why we considered this:** MaxPool always selects the maximum value in each 2×2 window, which is a hand-designed rule that discards three out of four values. Strided convolutions learn how to optimally combine nearby values during downsampling, potentially preserving more task-relevant spatial information. The tradeoff is more parameters and less inductive bias.

**Result:** 99.52% and the **fastest training time** (236.9 s). The accuracy difference vs. baseline is within the noise threshold (0.03 pp). On a dataset where MaxPool already works well, the fixed rule is sufficient and the added flexibility of learned downsampling provides no measurable benefit.

### 4.6 Parameter Sensitivity

To understand how sensitive our results were to key hyperparameters, we ran a Bayesian hyperparameter search using Optuna (Akiba et al., 2019) with a Tree-structured Parzen Estimator (TPE) across learning rate (1×10⁻⁴ to 1×10⁻²), dropout (0.2–0.6), batch size (32/64/128), optimizer (Adam/SGD), and weight decay (1×10⁻⁵ to 1×10⁻³).

| Hyperparameter | Search Range | Most Effective Region |
|---------------|-------------|----------------------|
| Learning rate | 1×10⁻⁴ to 1×10⁻² | 5×10⁻⁴ to 2×10⁻³ |
| Dropout rate | 0.2 – 0.6 | 0.3 – 0.5 |
| Batch size | 32, 64, 128 | All similar |
| Optimizer | Adam, SGD | Adam consistently better |
| Weight decay | 1×10⁻⁵ to 1×10⁻³ | Low end (1×10⁻⁵ to 1×10⁻⁴) |

The key finding is that the dataset is relatively insensitive to hyperparameter choices within a reasonable range: all Adam trials with learning rate between 5×10⁻⁴ and 2×10⁻³ reached similar accuracy. SGD trials were more sensitive and required careful tuning. Batch size had almost no effect on final accuracy, only on training speed. Dropout below 0.3 led to marginally higher validation loss. This exploratory search confirmed that our manually chosen defaults sit in a well-performing region of the hyperparameter space and that the results are not an artifact of a lucky configuration.

### 4.7 Latent Space Visualisation

To understand what the network learned internally, feature vectors were extracted from the penultimate layer of the baseline CNN and projected to two dimensions using t-SNE (van der Maaten & Hinton, 2008) with perplexity 30. If the 43 classes form distinct clusters in the 2D projection, the network has learned a representation where similar signs are close together and different signs are far apart, providing interpretable evidence beyond accuracy numbers alone.

### 4.8 Autoencoder for Anomaly Detection

A key limitation of any classifier is that it always assigns an input to one of its known classes, even when the input is entirely outside the training distribution. We implemented a convolutional autoencoder as a complementary anomaly detection mechanism, applying the concept from Lecture 7.

The encoder compresses 3×32×32 images through three convolutional blocks down to a 128-dimensional latent vector; a mirrored decoder with transposed convolutions reconstructs the original image. Training is fully unsupervised and minimises the per-pixel MSE between input and reconstruction:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{a}_i - a_i)^2$$

After training, reconstruction error serves as an anomaly score: known signs are reconstructed accurately (low error), while degraded or unknown inputs produce high reconstruction error. A threshold at the 95th percentile of the validation error distribution flags such inputs as anomalous. This component was implemented as a proof-of-concept for Lecture 7; quantitative evaluation on out-of-distribution samples was beyond the scope of this project.

---

## 5. Results and Discussion

The Deep CNN was selected as the best model and evaluated in depth on the held-out test set.

### 5.1 Test Set Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| Test Accuracy (Top-1) | **99.81%** | Share of test images where the model's first prediction is correct |
| Test Accuracy (Top-5) | **99.98%** | Share where the correct class appears in the top-5 predictions |
| Test Loss | 0.0061 | Average cross-entropy loss (lower is better; reflects prediction confidence) |
| Wrong Classifications | 11 / 5,881 | Absolute number of incorrect predictions on the test set |

The Top-5 accuracy of 99.98% means the correct class appears among the model's five most confident predictions in all but two test cases; even when the top prediction is wrong, the model almost always assigns high probability to the correct class.

### 5.2 Confusion Matrix

![Normalized confusion matrix of the Deep CNN on the test set](results/task06/deep/confusion_matrix_normalized.png)

A confusion matrix shows, for each true class (rows), how the model distributed its predictions across all classes (columns). Each cell in row *i*, column *j* contains the fraction of images truly belonging to class *i* that were predicted as class *j*. A perfect classifier produces a pure diagonal matrix where every image is predicted as its true class.

Our confusion matrix is strongly diagonal, meaning the model is almost always correct. The few visible off-diagonal entries are concentrated among visually similar sign pairs, for example different speed limit signs (30/50/80 km/h) that share circular shapes and differ only in the printed number, and warning signs with similar triangular layouts. These are precisely the hardest cases for any classifier operating at 32×32 resolution, where small numerical differences are difficult to resolve.

### 5.3 Per-Class Accuracy

![Per-class test accuracy across all 43 GTSRB classes](results/task06/deep/per_class_accuracy.png)

**Five best-performing classes (100% accuracy):** Stop, Dangerous curve left, Dangerous curve right, End of no passing, End of no passing by vehicles over 3.5t.

**Five worst-performing classes:**

| Class ID | Name | Test Accuracy | Likely Reason |
|----------|------|:---:|---------------|
| 27 | Pedestrians | 97.62% | Similar layout to class 18 (General caution) and class 26 (Traffic signals) |
| 29 | Bicycles crossing | 97.62% | Icon very similar to Pedestrians sign (class 27) |
| 21 | Double curve | 98.39% | Resembles single curve warning signs at low resolution |
| 30 | Beware of ice/snow | 98.67% | Snowflake detail difficult to resolve at 32×32 |
| 8  | Speed limit (120km/h) | 99.10% | "120" can be confused with "100" (class 7) at small sizes |

The pattern is clear: every underperforming class is visually similar to at least one neighbour. The Pedestrians and Bicycles crossing signs (classes 27 and 29) are particularly prone to confusion, as both are triangular warning signs with a human silhouette icon. At 32×32 pixels, the difference between a pedestrian and a cyclist silhouette is only a handful of pixels. This is an inherent limitation of the 32×32 input resolution, not a fundamental failure of the model.

### 5.4 Precision and Recall

![Precision and recall per class for the Deep CNN](results/task06/deep/precision_recall_per_class.png)

**Precision** measures how reliable the model is when it predicts a specific class: of all images the model labeled as class *X*, what fraction actually belongs to class *X*? Low precision means the model is generating many false positives for that class, confidently predicting signs that are actually something else.

**Recall** measures how complete the model's detection is: of all images that truly belong to class *X*, what fraction did the model correctly identify? Low recall means the model is missing many instances of that class, failing to recognize signs that were actually there.

Both metrics matter independently for traffic sign recognition. A model with high recall but low precision might correctly find all stop signs but also misclassify many other signs as stop signs, creating false alerts. A model with high precision but low recall might never raise a false alarm but miss real stop signs entirely, which could be dangerous in practice.

Our Deep CNN shows consistently high precision and recall across all 43 classes. The few classes with slightly reduced scores correspond exactly to the visually ambiguous categories identified in Section 5.3, confirming that the remaining errors are concentrated in genuinely hard cases rather than spread across all classes.

### 5.5 Misclassified Examples

![High-confidence misclassifications: cases where the model was wrong but confident](results/task06/deep/misclassifications_top_confidence.png)

The misclassification grid shows the 11 incorrectly predicted test images, sorted by the model's (incorrect) confidence. In most cases the error is understandable: degraded image quality, partial occlusion, or strong visual similarity to another class. Errors are concentrated in genuinely hard cases, not systematic failures of an entire category.

### 5.6 Bias Analysis

A critical concern for deployment is whether the model performs disproportionately worse on underrepresented classes, a form of class frequency bias. We evaluated this by comparing the 10 most frequent and 10 least frequent classes by training count.

![Mean test accuracy for frequent vs. rare traffic sign classes](results/task06/deep/bias_analysis_mean_accuracy.png)

*Blue bars (left): the 10 most frequent classes, each with 1,000+ training images. Orange bars (right): the 10 rarest classes, each with fewer than 210 training images. The dashed lines show the mean accuracy for each group. Training counts (n=...) are shown inside each bar label.*

| Group | Training images (avg.) | Mean Test Accuracy |
|-------|----------------------|-------------------|
| Frequent classes (top 10) | ~1,374 per class | 99.87% |
| Rare classes (bottom 10) | ~169 per class | 99.52% |
| Gap | N/A | **0.34 percentage points** |

The 0.34 pp gap between the most and least represented classes is remarkably small. Notably, several of the rarest classes, such as Speed limit (20 km/h) and Dangerous curve left (each with approximately 147 images in the training split), achieve 100% test accuracy. This suggests that the augmentation strategy and training procedure generalize well even for classes with very few examples, without requiring explicit oversampling or class weighting. These figures are based on a single training run and data split; the absolute gap may vary across runs. A model that is accurate on average but fails on rare classes would be unsuitable for deployment, since rare signs require reliable recognition precisely because they appear infrequently in real traffic.

### 5.7 Robustness Testing

In real-world deployment, camera images are rarely as clean as the GTSRB training data. We evaluated the Deep CNN under two standard image perturbations applied at inference time; the model was not retrained with these distortions, so the test measures how well clean-trained features generalize to degraded inputs.

| Condition | Test Accuracy | Δ vs. Clean | What this simulates |
|-----------|-------------|:-----------:|---------------------|
| Clean | 99.81% | baseline | Ideal conditions |
| Gaussian Blur (kernel=5) | 97.01% | −2.80 pp | Motion blur, out-of-focus optics, fog |
| Gaussian Noise (σ=0.1) | 71.86% | **−27.95 pp** | Low-quality sensors, compression artifacts |

The model handles blur well: a 2.80 pp drop is minor and expected, since blurring mainly smooths fine details that may not be critical for sign recognition. The **Gaussian noise result is far more concerning**: a 27.95 pp accuracy drop from a moderate noise level (σ=0.1 applied to normalized [0,1] pixel values) reveals a significant vulnerability. CNNs trained exclusively on clean images learn to rely on precise pixel-level patterns that break down immediately when random noise distorts those patterns. This is the most critical gap between benchmark performance and real-world reliability. The practical fix is straightforward: adding noise augmentation during training would address this directly, but was beyond the scope of this project.

### 5.8 Grad-CAM Interpretability

![Grad-CAM visualizations: image regions that most influenced the model's predictions](results/task06/deep/gradcam_examples.png)

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) is an interpretability technique that answers the question: *which parts of the input image did the model actually look at when making its prediction?* It works by computing the gradient of the predicted class score with respect to the activations of the final convolutional layer; regions with high gradient magnitude had the most influence on the prediction.

The visualizations provide evidence that the model consistently attends to the relevant sign regions, namely the shape, symbol, and color content, rather than background artifacts like sky, road, or surrounding objects. A model that relies on spurious correlations in the background would be fragile under any change of scene context. The Grad-CAM results suggest this is not the case here, giving us additional confidence that the model has learned meaningful visual representations.

### 5.9 Discussion

All five models exceed 99% test accuracy, which provides evidence that CNN-based classifiers are well-suited to the GTSRB task. The results largely matched our initial expectations. Adding a fourth convolutional block (Deep CNN) proved to be the most cost-effective improvement, gaining 0.32 percentage points over the baseline at nearly identical training time. The intuition was correct: additional depth allowed the network to learn more abstract feature representations that better separate visually similar classes. Transfer learning via MobileNetV2 delivered a smaller accuracy gain at four times the parameter count and twice the training time, confirming that the GTSRB training set is large enough for compact from-scratch CNNs to learn excellent representations without ImageNet pretraining. The results for Leaky ReLU and Stride CNN fell within the noise threshold, suggesting that BatchNorm is the dominant stabilizing factor and that the choice of activation function and downsampling method is secondary.

The bias analysis showed a 0.34 percentage point accuracy gap between the most and least frequent classes, which is a surprisingly small difference given the 11× class imbalance. The augmentation strategy and training procedure appear to generalize well even for the rarest classes. The Grad-CAM visualizations provide supporting evidence that predictions are based on sign-relevant features rather than background correlations. The noise robustness result was the one outcome that exceeded our expected severity: a 27.95 pp drop under moderate Gaussian noise is significant, and is the clearest gap between benchmark performance and real-world reliability. This is a well-known vulnerability of CNNs trained exclusively on clean data and would need to be addressed before any deployment in safety-critical scenarios.

### 5.10 Limitations

Several limitations should be noted when interpreting these results. All improved model variants were trained once with a single random seed and a fixed data split; performance estimates would be statistically more reliable with multiple independent runs and cross-validation. The 32×32 input resolution discards spatial detail, which likely explains the reduced accuracy on visually similar classes such as Pedestrians and Bicycles crossing. No noise or blur augmentation was applied during training, which directly explains the poor noise robustness identified in Section 5.7. Finally, the dataset itself introduces selection biases: GTSRB was recorded exclusively on German roads from a single camera system and contains no damaged or vandalized signs. These factors limit generalizability to other road environments and real-world deployment conditions.

---

## 6. Conclusion and Future Work

### 6.1 Conclusion

This project set out to evaluate how different CNN architectures perform on the GTSRB traffic sign classification task and to identify which design decisions drive performance improvements. Three central findings emerged.

The first finding is that a compact CNN trained from scratch is sufficient for this task. The baseline model, with 629K parameters, reaches 99.49% test accuracy on our internal hold-out split. Stallkamp et al. (2012) reported a human recognition rate of 98.84% on the official GTSRB test set; while this is not directly comparable to our internal evaluation setup, the baseline result lies in the same broad performance range. This outcome is not surprising: traffic signs are designed for fast reliable recognition using standardized shapes and colors, their visual structure is inherently suited to convolutional feature learning, and the GTSRB images are pre-cropped to the sign bounding box, reducing the task to pure classification.

The second finding is that increased depth was the only architectural change that produced a meaningful improvement. Adding a fourth convolutional block (Deep CNN) raised test accuracy to 99.81%, reducing wrong predictions from 30 to 11 out of 5,881. Replacing ReLU with Leaky ReLU and swapping MaxPooling for strided convolutions both produced changes within the noise threshold, suggesting that Batch Normalization is the dominant stabilizing factor and that the specific choice of activation function and downsampling strategy is secondary for this task. MobileNetV2 improved over the baseline but did not match the Deep CNN despite having four times the parameters and nearly twice the training time, indicating that ImageNet pretraining provides no clear advantage when the GTSRB training data already covers the target domain well.

The third finding concerns the gap between benchmark performance and real-world reliability. The bias analysis showed that the Deep CNN generalizes well across class frequency groups, with a 0.34 pp accuracy gap between frequent and rare classes despite an approximately 11-fold imbalance, and Grad-CAM confirms that predictions are grounded in sign-relevant visual features. However, accuracy drops by 27.95 percentage points under moderate Gaussian noise. The main remaining limitation therefore appears to be less about model capacity and more about training conditions: a model trained exclusively on clean images has not been exposed to the kinds of sensor noise encountered in real deployments.

From a software perspective, the current implementation can be extended toward a full detection-classification pipeline; the detailed class structure is provided in Appendix C.

### 6.2 Future Work

The limitations identified in Section 5.10 point directly to the most valuable next steps. In the near term, adding Gaussian noise and blur augmentation to the training pipeline would close the largest performance gap identified in this project. Increasing the input resolution from 32×32 to 64×64 pixels would address the spatial detail lost at the current size and is expected to reduce confusion between visually similar classes. Over the longer term, integrating an object detection stage would extend the system to uncropped road footage, and evaluating across multiple random splits would yield more reliable performance estimates for rare classes. Fine-tuning on traffic sign data from other countries and adverse weather conditions would reduce GTSRB's geographical and meteorological selection bias and improve generalizability to real-world deployment.

---

## References

Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32, 323–332. https://doi.org/10.1016/j.neunet.2012.02.016

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.

---

## Appendix A: Generated Artifacts

| Task | Key Output Files |
|------|-----------------|
| Task 02 | `results/class_mapping.csv`, `results/task03/class_distribution.png` |
| Task 03 | `results/preprocessing_stats.json`, `results/preprocessing_sample_grid.png` |
| Task 04 | `models/baseline.pth`, `results/task04/baseline_history_seed-42.json`, `results/task04/baseline_loss_curve_seed-42.png` |
| Task 05 | `models/deep_cnn.pth`, `results/task05/model_comparison.json`, `results/task05/model_comparison_summary.png` |
| Task 06 | `results/task06/deep/evaluation_summary.json`, `results/task06/deep/gradcam_examples.png`, `results/task06/deep/confusion_matrix_normalized.png`, `results/task06/deep/bias_analysis_mean_accuracy.png` |

---

## Appendix B: GTSRB Class ID Legend

| ID | Class Name | ID | Class Name | ID | Class Name | ID | Class Name |
|----|------------|----|------------|----|------------|----|------------|
| 0 | Speed limit (20 km/h) | 11 | Right-of-way at next intersection | 22 | Bumpy road | 33 | Turn right ahead |
| 1 | Speed limit (30 km/h) | 12 | Priority road | 23 | Slippery road | 34 | Turn left ahead |
| 2 | Speed limit (50 km/h) | 13 | Yield | 24 | Road narrows on the right | 35 | Ahead only |
| 3 | Speed limit (60 km/h) | 14 | Stop | 25 | Road work | 36 | Go straight or right |
| 4 | Speed limit (70 km/h) | 15 | No vehicles | 26 | Traffic signals | 37 | Go straight or left |
| 5 | Speed limit (80 km/h) | 16 | Vehicles over 3.5t prohibited | 27 | Pedestrians | 38 | Keep right |
| 6 | End of speed limit (80 km/h) | 17 | No entry | 28 | Children crossing | 39 | Keep left |
| 7 | Speed limit (100 km/h) | 18 | General caution | 29 | Bicycles crossing | 40 | Roundabout mandatory |
| 8 | Speed limit (120 km/h) | 19 | Dangerous curve left | 30 | Beware of ice/snow | 41 | End of no passing |
| 9 | No passing | 20 | Dangerous curve right | 31 | Wild animals crossing | 42 | End of no passing by vehicles over 3.5t |
| 10 | No passing for vehicles over 3.5t | 21 | Double curve | 32 | End of all speed and passing limits | | |

---

## Appendix C: Current Class Structure and Extension Points

All five classifier models share a common design: they inherit from PyTorch's `nn.Module` base class and expose a `features` extractor and a `classifier` head. The autoencoder follows the same base class but uses an encoder-decoder structure. The diagram below shows the current class hierarchy and marks the three components not yet implemented (tagged `future extension`):

```mermaid
classDiagram
    class nn_Module {
        <<PyTorch Base>>
        +forward(x) Tensor
    }

    class BaselineCNN {
        +features: Sequential
        +classifier: Sequential
        +forward(x) Tensor
    }

    class DeepCNN {
        +features: Sequential
        +classifier: Sequential
        +forward(x) Tensor
    }

    class LeakyReLUCNN {
        +features: Sequential
        +classifier: Sequential
        +forward(x) Tensor
    }

    class StrideCNN {
        +features: Sequential
        +classifier: Sequential
        +forward(x) Tensor
    }

    class MobileNetTransfer {
        +features: Sequential
        +classifier: Sequential
        +pool: AdaptiveAvgPool2d
        +forward(x) Tensor
    }

    class ConvAutoencoder {
        +encoder: Sequential
        +decoder: Sequential
        +threshold: float
        +forward(x) Tensor
        +is_anomaly(x) bool
    }

    class ObjectDetector {
        <<future extension>>
        +detect(frame) BoundingBoxes
    }

    class SignClassifier {
        <<future extension>>
        +model: nn_Module
        +classify(crop) ClassLabel
    }

    class FullPipeline {
        <<future extension>>
        +detector: ObjectDetector
        +classifier: SignClassifier
        +anomaly_filter: ConvAutoencoder
        +run(frame) Result
    }

    nn_Module <|-- BaselineCNN
    nn_Module <|-- DeepCNN
    nn_Module <|-- LeakyReLUCNN
    nn_Module <|-- StrideCNN
    nn_Module <|-- MobileNetTransfer
    nn_Module <|-- ConvAutoencoder
    nn_Module <|-- ObjectDetector
    FullPipeline o-- ObjectDetector
    FullPipeline o-- SignClassifier
    FullPipeline o-- ConvAutoencoder
    SignClassifier o-- nn_Module
```

`ObjectDetector` would wrap a detection model such as YOLO and locate sign bounding boxes in raw camera frames. `SignClassifier` would wrap any existing classifier and apply it to each detected crop. `FullPipeline` would compose all three components into a deployable system. Because all classifiers already share the same `forward(x)` interface, any of the five existing models can be plugged into `SignClassifier` without changes to the surrounding pipeline code.
