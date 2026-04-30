---
marp: true
theme: default
paginate: true
style: |
  :root {
    --color-accent: #c0392b;
    --color-dark:   #1a1a2e;
    --color-mid:    #2c3e50;
    --color-light:  #f4f6f7;
  }
  section {
    font-size: 21px;
    color: var(--color-dark);
    background: #ffffff;
  }
  h1 {
    color: var(--color-dark);
    font-size: 1.9em;
    border-bottom: 3px solid var(--color-accent);
    padding-bottom: 8px;
  }
  h2 {
    color: var(--color-mid);
    font-size: 1.5em;
  }
  h3 {
    color: var(--color-accent);
    font-size: 1.1em;
    margin-bottom: 4px;
  }
  strong { color: var(--color-accent); }
  blockquote {
    border-left: 4px solid var(--color-accent);
    background: var(--color-light);
    padding: 10px 18px;
    font-style: normal;
    font-size: 0.95em;
  }
  table { font-size: 0.82em; width: 100%; }
  th { background: var(--color-mid); color: #fff; }
  section.title {
    background: var(--color-dark);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 { color: #ffffff; border-color: var(--color-accent); font-size: 2em; }
  section.title h2 { color: #cccccc; font-size: 1.2em; }
  section.title p  { color: #aaaaaa; }
  section.divider {
    background: var(--color-mid);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.divider h1 { color: #ffffff; border-color: var(--color-accent); text-align: center; }
  section.image-focus {
    background: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding-left: 56px;
    padding-right: 56px;
  }
  section.image-focus h1,
  section.image-focus h2 {
    width: 100%;
    font-size: 1.25em;
    margin-bottom: 10px;
    text-align: center;
  }
  section.image-focus p {
    width: 100%;
    text-align: center;
  }
  section.image-focus img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 92%;
    max-height: 74vh;
    object-fit: contain;
  }
  section.image-focus strong {
    display: block;
    text-align: center;
  }
  .highlight-row { background: #fdebd0; font-weight: bold; }
  .kicker {
    color: var(--color-accent);
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .lead {
    font-size: 1.35em;
    font-weight: 700;
    line-height: 1.25;
    color: var(--color-dark);
    margin: 8px 0 18px 0;
  }
  .muted {
    color: #6c757d;
  }
  .cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 18px;
  }
  .card {
    background: var(--color-light);
    border-left: 5px solid var(--color-accent);
    border-radius: 10px;
    padding: 16px 18px;
    min-height: 110px;
  }
  .card h3 {
    margin-top: 0;
    margin-bottom: 8px;
  }
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 22px;
  }
  .kpi {
    background: #ffffff;
    border: 2px solid #e5e8e8;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
  }
  .kpi .number {
    color: var(--color-accent);
    font-size: 2.0em;
    font-weight: 800;
    line-height: 1.05;
  }
  .kpi .label {
    color: var(--color-mid);
    font-size: 0.78em;
    margin-top: 6px;
  }
  .takeaway {
    background: #fff5f2;
    border-left: 6px solid var(--color-accent);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 1.05em;
    margin-top: 16px;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
  }
  .step-label {
    display: inline-block;
    background: var(--color-dark);
    color: #ffffff;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72em;
    font-weight: 700;
    margin-right: 8px;
  }
  section.divider .subtitle {
    color: #d9d9d9;
    font-size: 1.0em;
    margin-top: 8px;
  }
---

<!-- _class: title -->
<!-- _paginate: false -->

<div class="kicker">Machine Learning Project | KLU 2026</div>

# CNN Traffic Sign Classification

## What actually improves performance on GTSRB?

German Traffic Sign Recognition Benchmark

<!--
Speaker notes:
Open with the core question, not just the dataset name. The presentation is about identifying which design decisions actually matter for traffic sign classification. Keep it brief -- the next slide explains why the problem matters.
Time: 0:30
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# 1. Why this problem matters

<div class="subtitle">Traffic signs are simple by design, but not always simple for a model.</div>

---

## Why This Matters

<div class="kicker">Problem framing</div>

<div class="lead">A traffic sign classifier must make reliable decisions from imperfect visual input.</div>

<div class="cards">
  <div class="card">
    <h3>Safety relevance</h3>
    <p>Speed limits, priority signs, and stop signs directly affect driving decisions.</p>
  </div>
  <div class="card">
    <h3>Visual degradation</h3>
    <p>Illumination, blur, distance, and occlusion reduce available information.</p>
  </div>
  <div class="card">
    <h3>Decision pressure</h3>
    <p>The system ultimately has to output one traffic sign class, not a vague suggestion.</p>
  </div>
</div>

<div class="takeaway">
<strong>Guiding question:</strong> Can a compact CNN trained from scratch reach very high accuracy, and which design choices actually matter?
</div>

<p class="muted">The reported human recognition rate on the official GTSRB benchmark is 98.84%. We use this as context, not as a directly comparable evaluation target.</p>

<!--
Speaker notes:
Motivate the problem briefly. Emphasise the safety relevance and the gap between clean benchmark images and real-world visual degradation. Be explicit that the human benchmark is context only, because our evaluation uses an internal hold-out split.
Time: 1:00
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# 2. The experiment setup

<div class="subtitle">A controlled dataset, a compact baseline, and four targeted model variants.</div>

---

## The GTSRB Dataset

<div class="kicker">Dataset foundation</div>

<div class="lead">GTSRB provides a controlled classification task with realistic visual variation.</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">39,209</div><div class="label">labelled images</div></div>
  <div class="kpi"><div class="number">43</div><div class="label">traffic sign classes</div></div>
  <div class="kpi"><div class="number">70/15/15</div><div class="label">internal split</div></div>
</div>

<br>

<div class="two-col">
  <div>
    <span class="step-label">Scope</span> Images are pre-cropped to the sign bounding box, so this is a classification task, not detection.
  </div>
  <div>
    <span class="step-label">Caveat</span> Official test labels were not available, so our results are not directly comparable to the GTSRB leaderboard.
  </div>
</div>

<!--
Speaker notes:
Introduce the dataset using the three key numbers. Highlight that images come pre-cropped -- the model only has to classify, not locate signs. Mention the evaluation caveat early and honestly: our split is internal. The comparison between our five models is still valid because all are evaluated under identical conditions.
Time: 1:00
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# 3. The model logic

<div class="subtitle">Start simple, then test one main design idea at a time.</div>

---

<!-- _class: image-focus -->

## Dataset Overview: Class Distribution

![](results/task03/class_distribution.png)

**Key point:** the dataset covers 43 classes, but the class frequencies are uneven.

<!--
Speaker notes:
Use this slide mainly visually. Point to the uneven bar heights and explain that this motivates the next slide on class imbalance. Keep this short.
Time: 0:25
-->

---

## Dataset Challenge 1: Class Imbalance

The 43 classes are **not equally represented**:

| | Count |
|---|---|
| Most frequent class (Speed limit 50 km/h) | **2,250 images** |
| Rarest classes (three classes) | **210 images each** |
| Imbalance ratio | **approx. 10.7x** |

After splitting:
- Rarest classes contribute roughly **147 training images** each
- Risk: weaker generalisation for underrepresented classes

<!--
Speaker notes:
Focus on the imbalance. 10.7x is substantial. Mention that the rarest classes get only about 147 training examples in the split. This motivates both the augmentation strategy and the bias analysis later in the presentation.
Time: 0:45
-->

---

## Dataset Challenge 2: Visual Ambiguity

**Within each class:** variation in brightness, contrast, and viewing angle

**Across classes:** many signs differ only in a **small internal detail**

- Speed limit signs: same circular shape, differ only in the numeral
- Warning signs: same triangular shape, differ only in the inner symbol

> At 32x32 pixels, these differences may span only a few pixels.

<!--
Speaker notes:
Introduce the visual ambiguity concept first. The next slide shows the sample grid in full size so the audience can actually see the similarities between classes.
Time: 0:25
-->

---

<!-- _class: image-focus -->

## Dataset Challenge 2: Sample Images

![](results/task03/sample_images_by_class.png)

**Key point:** many classes share the same outer shape and differ only in small internal details.

<!--
Speaker notes:
Use the sample grid to illustrate the inter-class similarity problem. Speed limits and triangular warning signs are the clearest examples. Do not explain every class. Use it as visual evidence for why the task is challenging even though accuracy later becomes very high.
Time: 0:35
-->

---

## Preprocessing and Augmentation

All images are resized to **32x32 pixels** before training.

**Training images** receive random augmentations on every pass:

| Augmentation | Simulates |
|---|---|
| Random rotation (up to 15 degrees) | Tilted camera angle |
| Brightness and contrast variation | Changing illumination |
| Small random image shifts | Off-centre framing |

**Validation and test images** receive only resize and normalisation -- no random changes.

<br>

Augmentation **increases effective training variety** and is especially beneficial for rare classes with few original images.

<!--
Speaker notes:
Keep this slide brief. Explain the principle: augmentation artificially creates variation so the model does not memorise exact pixel patterns. The key point is that val and test transforms are deterministic -- otherwise evaluation would be noisy.
Time: 1:00
-->

---

## Baseline CNN Architecture

**Three-stage feature extraction followed by classification**

- **Stage 1:** Extract low-level features (edges, colours, gradients)
- **Stage 2:** Combine into mid-level patterns (corners, curves, regions)
- **Stage 3:** Build high-level representations (shapes, symbols)
- **Classifier:** Map features to 43 traffic sign classes

<br>

At each stage: spatial resolution is reduced, the number of feature channels grows.

**629,291 trainable parameters** total.

<!--
Speaker notes:
Explain the architecture conceptually, not as code. The key idea is the hierarchy: simple patterns in early stages, complex ones later. The next slide shows the architecture diagram in full size.
Time: 1:00
-->

---

<!-- _class: image-focus -->

## Baseline vs. Deep CNN Architecture

![](results/diagrams/architecture_comparison.png)

**Key point:** the Deep CNN adds one extra feature extraction stage, increasing depth while keeping the model compact.

<!--
Speaker notes:
Use this slide to visually connect the baseline and Deep CNN. Point out that the Deep CNN is not a completely different model, but a controlled extension of the baseline. Keep the explanation conceptual.
Time: 0:30
-->

---


## Four Model Variants

Each variant targets **one main design aspect** relative to the baseline:

| Variant | What changes | Hypothesis |
|---|---|---|
| **Deep CNN** | Adds a fourth feature extraction stage | More depth captures finer detail |
| **MobileNetV2** | Uses a model pretrained on ImageNet | Transfer learning may help rare classes |
| **LeakyReLU CNN** | Changes the activation function | Prevents neurons from becoming permanently inactive |
| **Stride CNN** | Replaces fixed pooling with learned downsampling | Preserves more spatial information |

<br>

All five models trained under **identical conditions**: same optimiser, same data split, same augmentation, capped at 20 epochs.

<!--
Speaker notes:
Emphasise the controlled design: one change at a time. This makes it possible to attribute differences in performance to the specific change. Briefly explain the hypothesis behind each variant. The expectations were: depth should help most, activation and downsampling expected to be neutral, transfer learning expected to be competitive but expensive.
Time: 1:30
-->

---

## What Each Variant Tests

<div class="kicker">Model logic</div>

<div class="lead">The variants are not just different models. Each one tests a specific design assumption.</div>

<div class="cards">
  <div class="card">
    <h3>Depth</h3>
    <p>Can an extra feature extraction stage improve recognition of small symbols and numbers?</p>
  </div>
  <div class="card">
    <h3>Transfer learning</h3>
    <p>Does a pretrained image model help, or is GTSRB specific enough to learn from scratch?</p>
  </div>
  <div class="card">
    <h3>Training mechanics</h3>
    <p>Do activation choice or learned downsampling meaningfully change performance?</p>
  </div>
</div>

<div class="takeaway">
<strong>Purpose:</strong> each variant turns one modelling assumption into a measurable experiment.
</div>

<!--
Speaker notes:
Use this slide to make the model variants easier to understand conceptually. The Deep CNN tests whether more depth helps with small visual details. MobileNetV2 tests whether pretrained general visual features are useful. LeakyReLU and Stride CNN test whether training mechanics or downsampling choices matter. This prepares the audience for the result that only depth produced a meaningful improvement.
Time: 0:45
-->

<!-- _class: divider -->
<!-- _paginate: false -->

# 4. What changed performance?

<div class="subtitle">The answer was not bigger or pretrained. It was targeted depth.</div>

---

## Model Comparison: Results

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| Deep CNN | 99.81% | 11 | 936K | 284.0 s |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

<br>

All five models exceed **99% test accuracy** on the internal split.

<!--
Speaker notes:
Present the table without interpretation first. Let the audience absorb the numbers. Point out that all models are above 99% -- the baseline is already strong. Mention that 0.1 percentage points corresponds to roughly 6 images on 5,881, so small differences are not meaningful. The next slide highlights what matters.
Time: 0:30
-->

---

## Model Comparison: Best Result

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936K** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

<br>

The **Deep CNN** achieves the highest accuracy and the lowest error count.

<!--
Speaker notes:
Draw attention to the Deep CNN row. It has the most errors eliminated: 30 down to 11. The parameter count grows moderately (629K to 936K), and training time is almost unchanged (275 vs 284 seconds). This is the key result.
Time: 0:30
-->

---

## Model Comparison: Key Takeaway

<div class="kicker">Main result</div>

<div class="lead">The Deep CNN delivered the best accuracy-cost tradeoff.</div>

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936K** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

<div class="takeaway">
<strong>Interpretation:</strong> the Deep CNN reduced errors from 30 to 11 with moderate parameter growth and almost unchanged training time. MobileNetV2 improved over the baseline, but not efficiently.
</div>

<!--
Speaker notes:
Deliver the main message clearly. The Deep CNN is the best tradeoff. MobileNetV2 is the counterargument to transfer learning here: more expensive, less accurate than the Deep CNN. LeakyReLU and Stride CNN show that not every plausible architectural change is worth making.
Time: 1:00
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# 5. The benchmark gap

<div class="subtitle">High clean accuracy does not automatically mean real-world robustness.</div>

---

<!-- _class: image-focus -->

## Model Comparison: Visual Summary

![](results/task05/model_comparison_summary.png)

**Key point:** the Deep CNN gives the best accuracy-cost tradeoff.

<!--
Speaker notes:
Use the visual summary to reinforce the table result. Emphasize that the Deep CNN improves accuracy without a major increase in training time, while MobileNetV2 is much more expensive.
Time: 0:25
-->

---

## Best Model Evaluation: Deep CNN

**Test set performance:**

| Metric | Value |
|---|---|
| Top-1 accuracy | **99.81%** |
| Top-5 accuracy | **99.98%** |
| Wrong classifications | **11 out of 5,881** |

<br>

The Deep CNN was selected because it had the best accuracy-cost tradeoff.

Top-5 accuracy of 99.98% means: in all but **one** test case, the correct class was among the top 5 predictions.

<!--
Speaker notes:
Present the key metrics first. Do not go into the confusion matrix yet. The next slide shows it in full size so the audience can actually see the diagonal structure.
Time: 0:40
-->

---

<!-- _class: image-focus -->

## Confusion Matrix: Mostly Diagonal

![](results/task06/deep/confusion_matrix_normalized.png)

**Key point:** errors are rare and mostly occur between visually similar classes.

<!--
Speaker notes:
Use the confusion matrix visually. The important point is the strong diagonal, not the individual cell values. Explain that the few off-diagonal values indicate confusions between similar signs.
Time: 0:35
-->

---

## Best Model Evaluation: Where Errors Occur

**Five weakest classes:**

| Class | Accuracy | Why difficult |
|---|:---:|---|
| Pedestrians | 97.62% | Similar shape to General caution |
| Bicycles crossing | 97.62% | Near-identical silhouette to Pedestrians |
| Double curve | 98.39% | Looks like single curve at 32x32 |
| Beware of ice/snow | 98.67% | Fine snowflake detail lost at low resolution |
| Speed limit 120 km/h | 99.10% | "120" vs "100" differ by only a few pixels |

<br>

All remaining errors are **visually explainable** -- not random failures.

<!--
Speaker notes:
Show that the errors are concentrated in genuinely hard cases, not spread randomly. The next slide gives the per-class plot in full size.
Time: 0:40
-->

---

<!-- _class: image-focus -->

## Per-Class Accuracy

![](results/task06/deep/per_class_accuracy.png)

**Key point:** the weaker classes are visually similar to other signs, especially at 32x32 resolution.

<!--
Speaker notes:
Use the plot as visual support for the previous table. The exact values are less important than the pattern: most classes are near perfect, and the weaker ones have a visual explanation.
Time: 0:25
-->

---

## Bias Analysis: Frequent vs. Rare Classes

**Does the model perform worse on rare classes?**

| Group | Avg. training images | Mean test accuracy |
|---|:---:|:---:|
| 10 most frequent classes | approx. 1,374 | **99.87%** |
| 10 rarest classes | approx. 169 | **99.52%** |
| Gap | | **0.34 percentage points** |

<br>

- Only **0.34 pp** difference despite an **11x class imbalance**
- Several rare classes reach **100%** accuracy
- One training run, one fixed split -- this result should not be over-interpreted

<!--
Speaker notes:
The key message is reassuring: class imbalance did not translate into a large accuracy gap. But add the caution: with so few test images per rare class, a single mistake changes accuracy significantly. The next slide shows the bias plot in full size.
Time: 0:45
-->

---

<!-- _class: image-focus -->

## Bias Analysis: Visual Summary

![](results/task06/deep/bias_analysis_mean_accuracy.png)

**Key point:** frequent and rare classes perform similarly on this split.

<!--
Speaker notes:
Use the visual summary to reinforce the small gap. Point out that this does not prove there is no bias in general, but it suggests no strong class-frequency bias in this experiment.
Time: 0:25
-->

---

<!-- _class: image-focus -->

## Interpretability: Grad-CAM

![](results/task06/deep/gradcam_examples.png)

**Key point:** the model focuses on sign shape, symbol, and colour, not mainly on background.

<!--
Speaker notes:
Explain Grad-CAM simply: it highlights the regions that most influenced the prediction. The key observation is that the model looks at the sign itself, not the sky or road behind it. A model that relies on background patterns would be fragile in new scenes.
Time: 0:35
-->

---

<!-- _class: image-focus -->

## Interpretability: Latent Space Structure

![](results/tsne_feature_space.png)

**Key point:** most traffic sign classes form distinct clusters, while overlap remains mainly among visually similar classes.

<!--
Speaker notes:
Explain t-SNE briefly: it projects the model's internal representations into 2D so we can inspect whether classes are separated. Distinct clusters mean the model has learned class-discriminative features. Overlap among speed limit signs and warning signs is consistent with the error analysis.
Time: 0:35
-->

---

## Robustness Testing: Results

The model was tested with two common image distortions **without any retraining**:

| Condition | Test Accuracy | Drop |
|---|---:|---:|
| Clean images | 99.81% | baseline |
| Gaussian Blur (5x5 kernel) | 97.01% | -2.80 pp |
| Gaussian Noise | 71.86% | **-27.95 pp** |

<br>

The model was trained **only on clean images**, so this test reveals how well it generalises to degraded inputs.

<!--
Speaker notes:
Present the full table first. Blur is handled reasonably -- only 2.8 pp drop. But noise is a serious problem. Let that number sit for a moment before moving to the next slide.
Time: 0:30
-->

---

## Robustness Testing: The Noise Problem

| Condition | Test Accuracy | Drop |
|---|---:|---:|
| Clean images | 99.81% | baseline |
| Gaussian Blur (5x5 kernel) | 97.01% | -2.80 pp |
| **Gaussian Noise** | **71.86%** | **-27.95 pp** |

<br>

Under moderate Gaussian noise, accuracy drops from **99.81%** to **71.86%**. This corresponds to roughly **1,600 additional errors** on the same test set.

> **Clean benchmark accuracy does not automatically mean real-world robustness.**

<!--
Speaker notes:
Highlight the noise row clearly. A 27.95 pp drop is large. Explain why: the model was never exposed to noisy images during training. This is a classic distribution shift problem. The blur result is actually encouraging -- the model has learned something more abstract than just sharp pixels. But noise adds high-frequency variation that the model has never seen.
Time: 0:45
-->

---

## Robustness Testing: Key Message and Limitations

> **The main remaining challenge is not clean-image classification. It is robustness under distribution shift.**

<br>

**Study limitations to keep in mind:**

- Results are based on an **internal hold-out split**, not the official GTSRB test set
- All images are **pre-cropped** -- the model cannot locate signs in full road scenes
- Each model was trained **only once** with a fixed split -- small accuracy differences may not be stable
- Input resolution of **32x32 pixels** may remove fine visual details
- The model was never trained on **noisy or blurred inputs**
- GTSRB was recorded on **German roads with a single camera** under normal weather conditions

<!--
Speaker notes:
Be honest about the limitations. The internal split caveat is important -- we cannot compare directly to the official leaderboard. The pre-cropping caveat sets clear scope: this is a classification system, not a detection system. The resolution and noise points motivate the future work on the next slide.
Time: 0:30
-->

---

## Conclusion and Future Work

<div class="kicker">Final message</div>

<div class="lead">For GTSRB, the bottleneck is no longer clean-image classification. It is robustness.</div>

<div class="cards">
  <div class="card">
    <h3>1. Compact works</h3>
    <p>Baseline CNN reaches 99.49% on the internal test split.</p>
  </div>
  <div class="card">
    <h3>2. Depth helps</h3>
    <p>Deep CNN improves to 99.81% and reduces errors from 30 to 11.</p>
  </div>
  <div class="card">
    <h3>3. Noise hurts</h3>
    <p>Accuracy drops by 27.95 pp under Gaussian noise.</p>
  </div>
</div>

<div class="takeaway">
<strong>Next steps:</strong> train with noise and blur augmentation, test higher input resolution, and repeat the comparison across multiple seeds and splits.
</div>

<!--
Speaker notes:
Summarise the presentation as a story. The baseline already works well, depth is the only clear architectural improvement, and robustness is the real practical weakness. End with concrete next steps rather than a generic future-work list.
Time: 1:15
-->

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

## Questions?

<!--
Speaker notes:
Leave time for questions. If asked about implementation details, keep answers conceptual. If asked about the worst-performing classes, refer to the Pedestrians and Bicycles crossing example. If asked about future work, mention noise augmentation as the single highest-priority next step.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Backup Slides

---

## Backup: Autoencoder for Anomaly Detection

**Problem:** A standard classifier always assigns one of its 43 known classes -- even for inputs it has never seen.

**Approach:** Train a separate compression model that learns to reconstruct known traffic signs.

- Compress each image to a compact 128-dimensional representation
- Reconstruct the image from that representation
- Measure the **reconstruction error**
- Images with **high error** are flagged as potentially anomalous

| Metric | Value |
|---|---|
| Final validation loss | 0.5118 |
| Anomaly threshold (95th percentile) | 1.091 |
| Images flagged on validation set | 294 out of 5,881 (5.0%) |

> This is a **proof of concept** -- no true out-of-distribution test set was available to validate detection capability.

<!--
Speaker notes:
Explain the autoencoder idea simply: if the model has only learned to compress and reconstruct known traffic signs, then an unknown input will produce a poor reconstruction and therefore a high error. The threshold is set so that 5% of known signs are flagged -- a baseline false positive rate. Without out-of-distribution data we cannot measure true detection performance.
-->

---

## Backup: Hyperparameter Sensitivity (Optuna)

**Goal:** Verify that our manually chosen training settings are in a robust region.

Method: Bayesian search over 30 trials, each trained for 10 epochs on the Stride CNN.

| Hyperparameter | Search range | Best value found |
|---|---|:---:|
| Learning rate | 0.0001 to 0.01 | **0.00124** |
| Dropout rate | 0.2 to 0.6 | **0.274** |
| Batch size | 32, 64, or 128 | **32** |
| Optimiser | Adam or SGD | **Adam** |
| Weight decay | very small range | **0.000698** |

**Best trial:** 99.91% validation accuracy

- Adam dominated all top-5 trials
- Best learning rate (0.00124) is very close to our default (0.001)
- Optimal dropout (0.274) is lower than our default (0.5)

> Our default settings are in a reasonable region of the search space.

<!--
Speaker notes:
Explain Optuna briefly: it is a Bayesian search method that uses results from earlier trials to guide the next ones, rather than searching randomly. The key takeaway is reassuring: our defaults were not lucky guesses. Adam's dominance is also informative -- SGD was not competitive here.
-->

---

## Backup: Full Limitations

| Limitation | Impact |
|---|---|
| Internal hold-out split only | Results not directly comparable to official GTSRB leaderboard |
| Pre-cropped images | System cannot locate signs in full road scenes |
| Single training run per model | Small accuracy differences may not be stable across seeds |
| 32x32 input resolution | Fine details (numerals, small symbols) may be lost |
| No noise or blur augmentation during training | Model not robust to degraded inputs |
| GTSRB recorded on German roads, single camera, normal weather | Limited generalisation to other regions, conditions, or sign standards |
| Rare classes have few test images | A single missed prediction changes their per-class accuracy significantly |

<!--
Speaker notes:
Use this slide if the audience asks for more detail about the limitations. Each row corresponds to a concrete, actionable insight. The most important ones for a follow-up project are: noise augmentation, higher resolution, and multiple seeds.
-->

---

## Backup: Timing Plan

| Slide | Topic | Time |
|---|---|:---:|
| 1 | Title | 0:30 |
| 2 | Divider: Why this problem matters | 0:10 |
| 3 | Problem and Motivation | 1:00 |
| 4 | Divider: The experiment setup | 0:10 |
| 5 | Dataset Overview | 1:00 |
| 6 | Class Distribution (image) | 0:25 |
| 7 | Class Imbalance | 0:45 |
| 8 | Visual Ambiguity | 0:25 |
| 9 | Sample Images (image) | 0:35 |
| 10 | Preprocessing | 1:00 |
| 11 | Divider: The model logic | 0:10 |
| 12 | Baseline Architecture | 1:00 |
| 13 | Baseline vs Deep CNN (image) | 0:30 |
| 14 | Model Variants | 1:30 |
| 15 | What Each Variant Tests | 0:45 |
| 16 | Divider: What changed performance? | 0:10 |
| 17 | Model Comparison | 0:30 |
| 18 | Model Comparison: Best | 0:30 |
| 19 | Model Comparison: Key Takeaway | 1:00 |
| 20 | Model Comparison: Visual Summary (image) | 0:25 |
| 21 | Best Model Evaluation | 0:40 |
| 22 | Confusion Matrix (image) | 0:35 |
| 23 | Where Errors Occur | 0:40 |
| 24 | Per-Class Accuracy (image) | 0:25 |
| 25 | Bias Analysis | 0:45 |
| 26 | Bias Analysis: Visual Summary (image) | 0:25 |
| 27 | Grad-CAM (image) | 0:35 |
| 28 | Latent Space Structure (image) | 0:35 |
| 29 | Divider: The benchmark gap | 0:10 |
| 30 | Robustness: Results | 0:30 |
| 31 | Robustness: Noise Problem | 0:45 |
| 32 | Robustness: Key Message | 0:30 |
| 33 | Conclusion and Future Work | 1:15 |
| 34 | Thank You | -- |
| **Total** | | **~15:00** |
