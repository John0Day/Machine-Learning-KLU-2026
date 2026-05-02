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
    font-size: 1.45em;
    margin-bottom: 12px;
  }
  h3 {
    color: var(--color-accent);
    font-size: 1.05em;
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
  table { font-size: 0.82em; width: 100%; border-collapse: collapse; }
  th { background: var(--color-mid); color: #fff; padding: 6px 10px; }
  td { padding: 5px 10px; border-bottom: 0.5px solid #e0e0e0; }
  tr:last-child td { border-bottom: none; }
  section.title {
    background: var(--color-dark);
    color: #ffffff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding-left: 72px;
  }
  section.title h1 { color: #ffffff; border-color: var(--color-accent); font-size: 2.1em; }
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
  section.divider h1 { color: #ffffff; border-color: var(--color-accent); text-align: center; font-size: 2.2em; }
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
  section.image-focus h2 {
    width: 100%;
    font-size: 1.2em;
    margin-bottom: 10px;
    text-align: center;
  }
  section.image-focus p { width: 100%; text-align: center; }
  section.image-focus img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 92%;
    max-height: 72vh;
    object-fit: contain;
  }
  .highlight-row { background: #fdebd0; font-weight: bold; }
  .kicker {
    color: var(--color-accent);
    font-size: 0.70em;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .lead {
    font-size: 1.3em;
    font-weight: 700;
    line-height: 1.25;
    color: var(--color-dark);
    margin: 6px 0 16px 0;
  }
  .muted { color: #6c757d; }
  .cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 16px;
  }
  .cards-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
    margin-top: 16px;
  }
  .card {
    background: var(--color-light);
    border-left: 5px solid var(--color-accent);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .card h3 { margin-top: 0; margin-bottom: 6px; }
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 20px;
  }
  .kpi {
    background: #ffffff;
    border: 2px solid #e5e8e8;
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
  }
  .kpi .number {
    color: var(--color-accent);
    font-size: 2.0em;
    font-weight: 800;
    line-height: 1.05;
  }
  .kpi .label {
    color: var(--color-mid);
    font-size: 0.75em;
    margin-top: 5px;
  }
  .takeaway {
    background: #fff5f2;
    border-left: 6px solid var(--color-accent);
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 0.97em;
    margin-top: 16px;
  }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
    margin-top: 14px;
  }
  .step-label {
    display: inline-block;
    background: var(--color-dark);
    color: #ffffff;
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 0.70em;
    font-weight: 700;
    margin-right: 7px;
  }
  section.divider .subtitle {
    color: #d0d6de;
    font-size: 1.0em;
    margin-top: 10px;
  }
  .big-stat {
    font-size: 3.5em;
    font-weight: 800;
    color: var(--color-accent);
    line-height: 1.0;
    margin: 10px 0 4px 0;
  }
  .big-stat-label {
    font-size: 0.9em;
    color: var(--color-mid);
  }
---

<!-- _class: title -->
<!-- _paginate: false -->

<div class="kicker">Machine Learning Project · KLU 2026</div>

# Compact CNN Architectures for Traffic Sign Classification

## An experimental comparison on GTSRB

<p style="margin-top: 18px; font-size: 0.9em;">German Traffic Sign Recognition Benchmark · 43 classes · 39,209 images</p>

<!--
This project compares five CNN architectures for traffic sign classification on GTSRB. We will cover the problem structure, our experimental setup, what the results show, and where the key limitations are.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# The Problem

<div class="subtitle">A large real-world dataset with two structural challenges built in.</div>

<!--
Before the models, a brief look at what makes this task harder than it first appears.
-->

---

## GTSRB: a real-world classification benchmark with structural challenges

<div class="kicker">Dataset</div>

<div class="lead">39,209 labelled images across 43 classes — captured from a car-mounted camera on real German roads.</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">39,209</div><div class="label">labelled images</div></div>
  <div class="kpi"><div class="number">43</div><div class="label">sign classes</div></div>
  <div class="kpi"><div class="number">10.7×</div><div class="label">max class imbalance</div></div>
</div>

<div class="two-col" style="margin-top: 20px;">
  <div><span class="step-label">Scope</span> Images are pre-cropped to the sign boundary. This is a pure classification task — not detection.</div>
  <div><span class="step-label">Reference</span> The reported human recognition rate on GTSRB is <strong>98.84%</strong>. We use this as a broad performance reference, not a direct comparison target.</div>
</div>

<!--
GTSRB provides pre-cropped sign images — the model only needs to classify, not locate. The human benchmark of 98.84% is a published reference point. Our evaluation uses an internal split, so direct numerical comparison to that figure has limits.
-->

---

## Two structural challenges compound the difficulty of this task

<div class="kicker">Problem structure</div>

<div class="cards">
  <div class="card">
    <h3>Class imbalance</h3>
    <p>The most frequent class has <strong>2,250</strong> images. The rarest have <strong>210</strong>. After our 70/15/15 split, the rarest classes train on roughly <strong>147 examples</strong> each.</p>
  </div>
  <div class="card">
    <h3>Visual ambiguity</h3>
    <p>Speed limit signs share the same circular shape and differ only by numeral. At 32×32 pixels, "80" and "30" are nearly indistinguishable.</p>
  </div>
  <div class="card">
    <h3>Within-class variation</h3>
    <p>Lighting, blur, contrast, and camera angle vary within each class. The model must generalise across these conditions from limited training data.</p>
  </div>
</div>

<div class="takeaway">
<strong>Implication:</strong> high accuracy on a clean benchmark is a necessary condition but not a sufficient one. The harder problems are rare-class performance, visual ambiguity at low resolution, and input degradation.
</div>

<!--
Three challenges compound here. Rare classes have very little training data. Visual ambiguity means some distinctions are at the limit of what 32×32 resolution can convey. And within-class variation requires generalisation, not pattern memorisation.
-->

---

<!-- _class: image-focus -->

## Class distribution: the most frequent class has more than 10× the training data of the rarest

![](results/task03/class_distribution.png)

**After splitting, the rarest classes train on roughly 147 images each — compared to over 1,500 for the most frequent.**

<!--
The imbalance is immediately visible. Speed limit 50 dominates on the left; the rare classes on the right have a fraction of the data. Whether this gap translates into a per-class accuracy gap is something we test explicitly later.
-->

---

<!-- _class: image-focus -->

## At 32×32 pixels, numerals and similar warning symbols are the main source of visual ambiguity

![](results/task03/sample_images_by_class.png)

**This is the actual input the model receives. Speed limit signs and warning triangles with similar silhouettes are the hardest classes to distinguish.**

<!--
This shows what the model works with. The circular speed limit signs are nearly indistinguishable at this resolution — a direct constraint on achievable accuracy for those class pairs.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Our Approach

<div class="subtitle">One baseline. Four targeted experiments. One variable changed at a time.</div>

<!--
The key design principle: isolate one architectural variable per experiment so that every result is directly interpretable.
-->

---

## Each variant tests one independent design assumption

<div class="kicker">Experimental design</div>

<div class="lead">Changing one variable at a time ensures that observed differences can be attributed to a specific cause.</div>

<div class="cards">
  <div class="card">
    <h3>More depth</h3>
    <p>Does a fourth extraction stage improve recognition of fine numerals and symbols at 32×32?</p>
  </div>
  <div class="card">
    <h3>Transfer learning</h3>
    <p>Does ImageNet pretraining give a meaningful advantage on a focused, domain-specific dataset?</p>
  </div>
  <div class="card">
    <h3>Training mechanics</h3>
    <p>Do activation function or downsampling strategy meaningfully affect what the model learns?</p>
  </div>
</div>

<div class="takeaway">
<strong>Design rationale:</strong> without this isolation, any accuracy difference could originate from multiple sources simultaneously. With it, each result maps to a single cause.
</div>

<!--
A controlled design is necessary to draw conclusions. If multiple variables change at once, we cannot attribute observed differences to specific causes. Each design question becomes its own experiment, and a null result is itself informative.
-->

---

## The baseline processes images in three hierarchical stages

<div class="kicker">Baseline architecture — 629K parameters</div>

<div class="lead">Each stage extracts increasingly abstract representations from the input image.</div>

<div class="cards">
  <div class="card">
    <h3><span class="step-label">Stage 1</span> Low-level features</h3>
    <p>32 filters detect edges, colour boundaries, and brightness gradients. Spatial resolution halved via pooling.</p>
  </div>
  <div class="card">
    <h3><span class="step-label">Stage 2</span> Structural patterns</h3>
    <p>64 filters combine Stage 1 outputs into corners, curves, and sign contours. Resolution halved again.</p>
  </div>
  <div class="card">
    <h3><span class="step-label">Stage 3</span> Sign representations</h3>
    <p>128 filters assemble sign-level features — numerals in circles, symbols in triangles. Output: a 2,048-value feature vector.</p>
  </div>
</div>

<div class="takeaway">
<strong>Classifier:</strong> the 2,048-value vector is passed to a fully connected layer that maps it to one of 43 classes. The full pipeline is trained end to end on GTSRB.
</div>

<!--
Three convolutional stages build progressively abstract representations — from raw pixel gradients to sign-level features. The 2,048-dimensional output vector is then classified. This is the reference architecture that all four variants modify.
-->

---

## Each variant targets one specific question about the baseline

<div class="kicker">Variant motivation</div>

| Variant | Hypothesis tested | Change made |
|---|---|---|
| **Deep CNN** | 3 stages may not resolve fine numerals at 32×32 | Added a 4th stage with 256 filters (+307K params) |
| **MobileNetV2** | Scratch training on 39K images may underfit rare classes | Replaced network with one pretrained on 1.2M images |
| **LeakyReLU CNN** | ReLU may cause dead neurons and limit gradient flow | Replaced ReLU activation with LeakyReLU |
| **Stride CNN** | MaxPooling may discard spatially relevant detail | Replaced pooling with learnable stride-2 convolution |

<div class="takeaway">
<strong>Each variant is a testable hypothesis.</strong> A null result — no improvement — is also informative: it indicates the baseline already handles that aspect adequately.
</div>

<!--
The variants are not arbitrary — each targets a known limitation or open question. The table makes the hypothesis and the change explicit. I will now show which ones held up.
-->

---

## All five models were trained under identical conditions

<div class="kicker">Training setup</div>

<div class="two-col">
  <div>
    <p><strong>Augmentation — training set only</strong></p>
    <ul>
      <li>Rotation ±15° — simulates tilted camera angle</li>
      <li>Brightness and contrast variation</li>
      <li>Small translation shifts</li>
    </ul>
    <p style="margin-top: 10px;" class="muted">Validation and test: resize and normalise only. No stochastic transforms — evaluation is fully reproducible.</p>
  </div>
  <div>
    <p><strong>Fixed across all models</strong></p>
    <ul>
      <li>Optimiser: Adam, lr = 0.001</li>
      <li>Split: 70 / 15 / 15, seed 42</li>
      <li>Max 20 epochs with early stopping</li>
      <li>Test set: 5,881 images, fixed order</li>
    </ul>
  </div>
</div>

<!--
Identical training conditions are what make the architectural comparison valid. Any accuracy difference between models is attributable to architecture alone. Augmentation is applied to training only — evaluation conditions are fixed and reproducible.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Results

<div class="subtitle">All models reached high accuracy on the internal split. One stood out on the cost–accuracy tradeoff.</div>

<!--
Here are the results. I will highlight the main pattern, then show the two most instructive comparisons in detail.
-->

---

## All five models achieve above 99% accuracy on the internal test split

<div class="kicker">Full results</div>

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| Deep CNN | 99.81% | 11 | 936K | 284.0 s |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

<br>

All five models achieve above **99%** accuracy on this internal split. For context, the published human benchmark on GTSRB is **98.84%** — though direct comparison is limited by our internal evaluation setup.

<p class="muted" style="font-size: 0.85em; margin-top: 8px;">All results from single training runs (seed 42). Differences below ~0.1 pp (≈ 6 images on this test set) should be treated as potentially unstable without multi-seed validation.</p>

<!--
Every model exceeds 99% — the task is tractable for compact CNNs on this benchmark. Differences are small in absolute terms, so I will focus on the result that stands out in both error count and cost efficiency.
-->

---

## One additional extraction stage reduced errors by 63% at minimal added cost

<div class="kicker">Main finding</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.81%</div><div class="label">Deep CNN — test accuracy</div></div>
  <div class="kpi"><div class="number">11</div><div class="label">wrong predictions / 5,881</div></div>
  <div class="kpi"><div class="number">+8 s</div><div class="label">extra training time vs. baseline</div></div>
</div>

<div class="two-col" style="margin-top: 16px;">
  <div>
    <p>The Deep CNN adds a single 4th convolutional stage (256 filters) to the baseline. Wrong predictions dropped from <strong>30 to 11</strong> — a 63% reduction in errors.</p>
    <p style="margin-top: 10px;">Parameter count increased from 629K to <strong>936K</strong>. Training time increased by just 8 seconds.</p>
  </div>
  <div>
    <p>All other variants either matched or fell below baseline accuracy while adding computational cost.</p>
    <p style="margin-top: 10px;">Depth was the only design change that produced a consistent accuracy improvement on this dataset and split.</p>
    <p class="muted" style="font-size: 0.85em; margin-top: 8px;">Single run — multi-seed validation would be needed to confirm the magnitude of this advantage.</p>
  </div>
</div>

<!--
The 63% error reduction from one additional stage is the strongest signal in the results. The cost is negligible. All other variants showed no consistent improvement over the baseline — depth was the only effective variable.
-->

---

## MobileNetV2 showed no accuracy advantage over the Deep CNN at significantly higher cost

<div class="kicker">Transfer learning comparison</div>

<div class="two-col">
  <div>
    <p><strong>MobileNetV2</strong></p>
    <ul>
      <li>Pretrained on ImageNet — 1.2M images</li>
      <li><strong>2.56M parameters</strong> — about 4× the baseline, 2.7× the Deep CNN</li>
      <li><strong>518.7 s training</strong> — 1.8× the Deep CNN</li>
      <li>Test accuracy: <strong>99.66%</strong></li>
    </ul>
  </div>
  <div>
    <p><strong>Deep CNN</strong></p>
    <ul>
      <li>Trained from scratch on GTSRB only</li>
      <li><strong>936K parameters</strong></li>
      <li><strong>284.0 s training</strong></li>
      <li>Test accuracy: <strong>99.81%</strong></li>
    </ul>
  </div>
</div>

<div class="takeaway">
<strong>Interpretation:</strong> for this task and data volume, a purpose-built compact model outperformed the pretrained general model at lower cost. This result is specific to this setting — transfer learning may be more effective in lower-data regimes or with domain-adapted pretraining.
</div>

<!--
MobileNetV2 had a structural advantage: 1.2M pretraining images, 4× more parameters than the baseline. Yet the Deep CNN trained from scratch outperformed it here. The finding is bounded by this specific task and dataset — it does not generalise to transfer learning in general.
-->

---

<!-- _class: image-focus -->

## The Deep CNN achieves the best accuracy at near-baseline training cost

![](results/task05/model_comparison_summary.png)

**Higher parameter count and longer training time did not produce better accuracy in this experiment.**

<!--
This chart makes the cost–accuracy tradeoff visible across all five models. MobileNetV2 sits furthest right in cost, yet below the Deep CNN in accuracy. The Deep CNN is the Pareto-optimal choice in this comparison.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Error Analysis

<div class="subtitle">11 wrong predictions on 5,881 test images. A consistent pattern.</div>

<!--
With only 11 errors, we could inspect each one individually. Here is what they show.
-->

---

## Remaining errors show a consistent pattern linked to visual similarity

<div class="kicker">Error analysis — Deep CNN</div>

<div class="lead">All errors occur at class boundaries where two signs are near-identical at 32×32 pixels.</div>

| Example class | Per-class accuracy | Likely cause |
|---|:---:|---|
| Pedestrians | 97.62% | Same triangular shape as other warning signs |
| Bicycles crossing | 97.62% | Near-identical silhouette to Pedestrians at this resolution |
| Speed limit 120 km/h | 99.10% | "120" vs. "100" — few-pixel difference at 32×32 |

<div class="takeaway">
<strong>Interpretation:</strong> the error pattern is consistent with a resolution constraint — the model may lack sufficient pixel information to resolve certain class boundaries. This observation is based on visual inspection. Top-5 accuracy of 99.98% provides supporting evidence: the correct class appeared in the top 5 predictions in all but one test case.
</div>

<!--
Every error involves a class pair that looks nearly identical at 32×32. This is consistent with a resolution limitation rather than systematic model bias. Top-5 accuracy of 99.98% supports this interpretation — the correct answer was almost always among the model's most confident predictions.
-->

---

<!-- _class: image-focus -->

## The confusion matrix shows near-perfect diagonal with minimal off-diagonal activity

![](results/task06/deep/confusion_matrix_normalized.png)

**Top-5 accuracy: 99.98% — the correct class appeared in the model's top 5 predictions in all but one of 5,881 test cases.**

<!--
The strong diagonal confirms accurate classification across almost all classes. The few off-diagonal values cluster around visually similar pairs — consistent with the error pattern discussed on the previous slide.
-->

---

## The 11× class imbalance produced a 0.34 pp accuracy gap between frequent and rare classes

<div class="kicker">Bias analysis</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.87%</div><div class="label">mean accuracy — 10 most frequent classes</div></div>
  <div class="kpi"><div class="number">99.52%</div><div class="label">mean accuracy — 10 rarest classes</div></div>
  <div class="kpi"><div class="number">0.34 pp</div><div class="label">gap</div></div>
</div>

<br>

Several rare classes reach **100%** accuracy on the test split. The augmentation strategy appears to have partially compensated for the data shortage in rare classes.

<p class="muted" style="font-size: 0.88em; margin-top: 14px;">Caveat: one training run, one fixed split. Rare classes have very few test images — a single missed prediction shifts per-class accuracy significantly. This result is indicative, not conclusive without multi-seed validation.</p>

<!--
An 11-fold data imbalance produced only a 0.34 pp accuracy gap. This is a positive signal, but the caveat matters: rare classes have too few test images for a stable estimate. Multi-seed validation would be needed to confirm whether this gap is representative.
-->

---

<!-- _class: image-focus -->

## Grad-CAM indicates that predictions are driven by the sign region, not background context

![](results/task06/deep/gradcam_examples.png)

**Activation maps concentrate on the sign shape and internal symbol — consistent with task-relevant feature learning.**

<!--
Grad-CAM highlights which image regions most influenced each prediction. Activations concentrate on the sign itself rather than the surrounding scene. This is consistent with the model learning task-relevant features — though Grad-CAM provides supportive evidence, not a formal proof of interpretability.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Robustness

<div class="subtitle">Clean benchmark accuracy does not predict performance under input degradation.</div>

<!--
The results so far describe performance on clean, well-cropped test images. Here is what changes when that condition does not hold.
-->

---

## Gaussian noise causes a 27.95 pp accuracy drop — the primary practical limitation

<div class="kicker">Robustness test — no retraining</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.81%</div><div class="label">clean test images</div></div>
  <div class="kpi"><div class="number">97.01%</div><div class="label">Gaussian blur — −2.8 pp</div></div>
  <div class="kpi"><div class="number" style="color: #c0392b;">71.86%</div><div class="label">Gaussian noise — −27.95 pp</div></div>
</div>

<br>

The model tolerates moderate blur reasonably well. Under Gaussian noise, accuracy drops from **99.81%** to **71.86%** — an increase of **1,647 wrong predictions** on the same 5,881 test images.

<div class="takeaway">
<strong>Cause:</strong> this is a distribution shift failure. The model was trained exclusively on clean images and has no learned response to noise. The benchmark result and the noise result describe two different operating conditions — only one of which is realistic in practice.
</div>

<!--
Blur tolerance is actually encouraging — the model has learned features that are not entirely dependent on sharp pixel boundaries. The noise result is the key limitation: a 27.95 pp drop is large enough to matter in any real application. The cause is not model capacity — it is that noise was absent from the training distribution.
-->

---

## Key constraints and three concrete follow-up directions

<div class="kicker">Limitations and next steps</div>

<div class="two-col">
  <div>
    <p><strong>Constraints on these results</strong></p>
    <ul>
      <li>Internal split — not comparable to the official GTSRB leaderboard</li>
      <li>Pre-cropped images — classification only, no sign detection</li>
      <li>Single run per model — small differences may not be stable across seeds</li>
      <li>32×32 input resolution — fine detail is lost</li>
      <li>GTSRB: German roads, controlled weather conditions only</li>
    </ul>
  </div>
  <div>
    <p><strong>Three highest-priority next steps</strong></p>
    <ul>
      <li><strong>Add noise and blur augmentation during training</strong> — most direct fix for the robustness gap</li>
      <li><strong>Repeat with multiple seeds and splits</strong> — validate stability of the Deep CNN advantage</li>
      <li><strong>Test at 64×64 resolution</strong> — expected to reduce confusion in similar-class pairs</li>
    </ul>
  </div>
</div>

<!--
The most actionable limitation is the noise result — augmentation training directly addresses it without requiring a new architecture. Multi-seed validation and higher resolution are the two other priorities before drawing stronger conclusions from this comparison.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Conclusion

<div class="subtitle">Depth improved accuracy. Robustness under noise remains unresolved.</div>

<!--
Three findings to close with.
-->

---

## Three findings from this experiment

<div class="kicker">Summary</div>

<div class="cards">
  <div class="card">
    <h3>Compact CNNs are sufficient here</h3>
    <p>A 629K-parameter baseline trained from scratch reaches <strong>99.49%</strong> on this internal split — in the same broad performance range as the reported human benchmark of 98.84%.</p>
  </div>
  <div class="card">
    <h3>Depth was the effective variable</h3>
    <p>One additional extraction stage reduced errors by <strong>63%</strong> at near-zero added cost. Transfer learning, activation choice, and downsampling strategy showed no consistent improvement.</p>
  </div>
  <div class="card">
    <h3>Robustness is the open problem</h3>
    <p>A <strong>27.95 pp</strong> drop under Gaussian noise shows that clean benchmark accuracy does not reflect degraded-input performance. Distribution shift is the primary unresolved challenge.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 20px;">
<strong>The benchmark result is strong on clean images. Extending it to degraded or out-of-distribution conditions requires explicit robustness work — the most important next step for this architecture.</strong>
</div>

<!--
Compact CNNs work well on GTSRB under controlled conditions. Depth was the only design change that consistently improved results. The noise result carries the most practical weight — it defines what this architecture still cannot do, and points directly to where future work should focus.
-->

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

## Questions?

<!--
Thank you. Backup slides are available on the autoencoder extension, hyperparameter sensitivity analysis, full limitations table, and t-SNE feature space visualisation.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Backup Slides

---

## Backup: Anomaly Detection via Autoencoder

<div class="kicker">Extension — proof of concept</div>

<div class="lead">A standard classifier always assigns one of its 43 known classes — even for inputs it has never seen.</div>

<div class="two-col">
  <div>
    <p><strong>Approach</strong></p>
    <p>A compression autoencoder learns to reconstruct known traffic signs via a 128-dimensional bottleneck. Unfamiliar inputs do not fit the learned compression pattern — they reconstruct poorly, producing elevated reconstruction error as an anomaly signal.</p>
    <br>
    <p><strong>Threshold</strong></p>
    <p>Set at the 95th percentile of validation reconstruction errors (1.091). This defines a 5% false positive rate on in-distribution data.</p>
  </div>
  <div>
    <p><strong>Results on the validation set</strong></p>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Latent size</td><td>128 dimensions</td></tr>
      <tr><td>Final reconstruction loss</td><td>0.5118</td></tr>
      <tr><td>Anomaly threshold (95th pct.)</td><td>1.091</td></tr>
      <tr><td>Images flagged</td><td>294 / 5,881 (5%)</td></tr>
    </table>
    <br>
    <p class="muted" style="font-size: 0.88em;">No true out-of-distribution test set was available. The method is implementable — its real-world detection performance is not yet validated.</p>
  </div>
</div>

<!--
The autoencoder adds an "I don't know" capability alongside the classifier. The method works in principle but remains unvalidated without a real OOD test set. This is a proof of concept, not a production-ready component.
-->

---

## Backup: Hyperparameter Sensitivity (Optuna)

<div class="kicker">Extension — 30-trial Bayesian search</div>

**Goal:** verify that manually chosen training settings are not fragile.

| Hyperparameter | Search range | Best value found |
|---|---|:---:|
| Learning rate | 0.0001 – 0.01 | **0.00124** |
| Dropout rate | 0.2 – 0.6 | **0.274** |
| Batch size | 32, 64, 128 | **32** |
| Optimiser | Adam / SGD | **Adam** |
| Weight decay | 1×10⁻⁵ – 1×10⁻² | **0.000698** |

**Best trial (trial 6):** 99.91% validation accuracy

- Adam dominated all top-5 trials — SGD was not competitive on this task
- Best learning rate (0.00124) is close to our default (0.001)
- Optimal dropout (0.274) is lower than our default (0.5)

<div class="takeaway">
<strong>Our default settings sit in a robust region of the search space.</strong>
</div>

<!--
Optuna uses earlier trial results to guide subsequent ones. The key finding: our manual defaults were not fragile. The optimal learning rate is very close to our choice. The one notable gap is dropout — lower regularisation appears preferable.
-->

---

## Backup: Full Limitations

| Limitation | Impact |
|---|---|
| Internal hold-out split only | Not directly comparable to official GTSRB leaderboard |
| Pre-cropped images | Cannot locate signs — classification only, not detection |
| Single training run per model | Small accuracy differences may not be stable across seeds |
| 32×32 input resolution | Fine details (numerals, symbols) may be partially lost |
| No noise or blur augmentation during training | Model not robust to degraded inputs |
| GTSRB: German roads, single camera, controlled weather | Limited generalisation to other regions, conditions, or sign standards |
| Rare classes have few test images | A single missed prediction shifts per-class accuracy significantly |

<!--
Use this slide for detailed questions about methodology. The most actionable limitations for follow-up work are noise augmentation, multi-seed validation, and higher resolution.
-->

---

<!-- _class: image-focus -->

## Backup: t-SNE Feature Space (Deep CNN)

![](results/tsne_feature_space.png)

**2,000 validation samples projected to 2D from the Deep CNN's 512-dimensional internal feature space (perplexity = 30).**

Most classes form distinct clusters. Overlap concentrates among speed limit signs and visually similar warning triangles — consistent with the per-class error pattern.

<!--
Distinct clusters indicate that the model has learned to separate classes in its internal representation space. The overlapping speed limit cluster directly explains where the remaining errors originate.
-->

---

## Backup: Timing Plan

| Slide | Topic | Time |
|---|---|:---:|
| 1 | Title | 0:20 |
| 2 | Divider — The Problem | 0:10 |
| 3 | Dataset overview | 0:50 |
| 4 | Two structural challenges | 0:45 |
| 5 | Class distribution (visual) | 0:20 |
| 6 | Sample images (visual) | 0:20 |
| 7 | Divider — Our Approach | 0:10 |
| 8 | Experimental design | 0:45 |
| 9 | Baseline architecture | 0:50 |
| 10 | Four variants | 1:00 |
| 11 | Training conditions | 0:40 |
| 12 | Divider — Results | 0:10 |
| 13 | Full results | 0:40 |
| 14 | Main finding: Deep CNN | 0:50 |
| 15 | Transfer learning comparison | 0:50 |
| 16 | Cost vs. accuracy (visual) | 0:25 |
| 17 | Divider — Error Analysis | 0:10 |
| 18 | Error pattern | 0:45 |
| 19 | Confusion matrix (visual) | 0:25 |
| 20 | Bias analysis | 0:45 |
| 21 | Grad-CAM (visual) | 0:25 |
| 22 | Divider — Robustness | 0:10 |
| 23 | Robustness results | 0:50 |
| 24 | Limitations and next steps | 0:50 |
| 25 | Divider — Conclusion | 0:10 |
| 26 | Summary | 0:55 |
| 27 | Thank You | — |
| **Total** | | **~14:45** |
