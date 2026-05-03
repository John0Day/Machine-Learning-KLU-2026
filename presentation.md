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
This project compares five CNN architectures for traffic sign classification on GTSRB. We will cover the problem structure, our experimental setup, the single-run and multi-seed results, and where the key limitations are.
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
| **LeakyReLU CNN** | ReLU may cause dead neurons and limit gradient flow | Replaced ReLU activation with LeakyReLU (slope = 0.01) |
| **Stride CNN** | MaxPooling may discard spatially relevant detail | Replaced pooling with learnable stride-2 convolution |

<div class="takeaway">
<strong>Each variant is a testable hypothesis.</strong> A null result — no improvement — is also informative: it indicates the baseline already handles that aspect adequately. Single-run results tell only part of the story; we validated all five models across three seeds.
</div>

<!--
The variants are not arbitrary — each targets a known limitation or open question. The table makes the hypothesis and the change explicit. Single-run results will be shown first, then multi-seed validation reveals which conclusions hold.
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

<div class="subtitle">All models exceed 99% on the internal split. Multi-seed analysis revised two of the initial rankings.</div>

<!--
Here are the results. I will first show the canonical single-run comparison, then what multi-seed validation added to the picture.
-->

---

## All five models achieve above 99% accuracy on the internal test split

<div class="kicker">Single-run results — canonical comparison (seed 42)</div>

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936K** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

<br>

All five models achieve above **99%** on this internal split. For context, the published human benchmark on GTSRB is **98.84%** — though direct comparison is limited by our internal evaluation setup.

<p class="muted" style="font-size: 0.85em; margin-top: 8px;">Canonical results from a single run (seed 42). Differences at or below ~0.1 pp (≈ 6 images) should not be interpreted without multi-seed context — see next slide.</p>

<!--
Every model exceeds 99% — the task is tractable for compact CNNs on this benchmark. Differences are small in absolute terms. The single-run ranking is a starting point; multi-seed validation shows which of these differences are reproducible.
-->

---

## One additional extraction stage reduced errors by 63% at minimal added cost

<div class="kicker">Main finding — Deep CNN</div>

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
    <p>Among all variants, depth produced the strongest accuracy–cost tradeoff. The Deep CNN advantage holds in multi-seed analysis: mean <strong>99.69% ± 0.17%</strong> across 3 seeds.</p>
    <p style="margin-top: 10px;" class="muted">The 0.18 pp mean gap over Baseline represents an average advantage — individual seed rankings vary. Seed 2026 is one example where Baseline outperformed Deep CNN.</p>
  </div>
</div>

<!--
The 63% error reduction from one additional stage is the strongest signal in the results. The cost is negligible. Multi-seed analysis confirms Deep CNN as the best model on average — but the advantage is not consistent in every single seed.
-->

---

## Multi-seed validation revised the ranking of two models from the single-run comparison

<div class="kicker">Stability analysis — 3 seeds × 5 models = 15 training runs</div>

| Rank | Model | Test Acc (mean ± std) | Parameters | Train Time (mean) |
|---:|---|---:|---:|---:|
| 1 | **Deep CNN** | **99.69% ± 0.17%** | 936K | 266 s |
| 2 | **LeakyReLU CNN** | **99.67% ± 0.03%** | 629K | 597 s |
| 3 | Baseline CNN | 99.51% ± 0.22% | 629K | 261 s |
| 4 | Stride CNN | 99.45% ± 0.12% | 823K | 268 s |
| 5 | MobileNetV2 | 99.43% ± 0.19% | 2.56M | 529 s |

<div class="two-col" style="margin-top: 12px;">
  <div>
    <p><strong>LeakyReLU CNN</strong> — seed 42 single-run: <strong>99.46%</strong> (last among CNNs). Multi-seed mean: <strong>99.67% ± 0.03%</strong> (2nd, smallest variance). The seed-42 result was an outlier. A cautionary example for single-run evaluation.</p>
  </div>
  <div>
    <p><strong>MobileNetV2</strong> — single-run: 99.66% (3rd). Multi-seed mean: <strong>99.43%</strong> (last). No stable accuracy advantage over purpose-built CNNs, despite ~4× more parameters and nearly twice the training time of the baseline.</p>
  </div>
</div>

<!--
Multi-seed validation is the methodological backbone of this project. Two models changed their story: LeakyReLU improved substantially; MobileNetV2 declined. Deep CNN's advantage held on average. The LeakyReLU reversal — from last to second — is the strongest argument for why single-run comparisons need to be interpreted with caution.
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

<p class="muted" style="font-size: 0.88em; margin-top: 14px;">Caveat: one training run, one fixed split. Rare classes have very few test images — a single missed prediction shifts per-class accuracy significantly. This result is indicative, not conclusive without cross-validation.</p>

<!--
An 11-fold data imbalance produced only a 0.34 pp accuracy gap. This is a positive signal, but the caveat matters: rare classes have too few test images for a stable estimate. The result suggests no strong frequency bias — but further validation with independent splits would be needed to confirm it.
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
      <li>Three-seed validation — not cross-validation or independent data splits</li>
      <li>32×32 input resolution — fine detail is lost</li>
      <li>GTSRB: German roads, controlled weather conditions only</li>
    </ul>
  </div>
  <div>
    <p><strong>Three highest-priority next steps</strong></p>
    <ul>
      <li><strong>Add noise and blur augmentation during training</strong> — most direct fix for the robustness gap</li>
      <li><strong>Validate across independent data splits</strong> — multi-seed is done; the next step is multiple splits to confirm ranking stability</li>
      <li><strong>Test at 64×64 resolution</strong> — expected to reduce confusion in visually similar class pairs</li>
    </ul>
  </div>
</div>

<!--
The most actionable limitation is the noise result — augmentation training directly addresses it without requiring a new architecture. Multi-seed validation has been completed; the remaining methodological gap is independent data splits. Higher resolution is the most plausible fix for the remaining per-class confusion errors.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Conclusion

<div class="subtitle">Depth was strongest. LeakyReLU surprised in multi-seed evaluation. Robustness remains open.</div>

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
    <h3>Architecture matters, but so does validation</h3>
    <p>Depth reduced errors by <strong>63%</strong> at near-zero cost — the strongest single change. Multi-seed analysis showed LeakyReLU CNN as second-best overall: the single-run result (last place) was a misleading outlier.</p>
  </div>
  <div class="card">
    <h3>Robustness is the open problem</h3>
    <p>A <strong>27.95 pp</strong> drop under Gaussian noise shows that clean benchmark accuracy does not reflect degraded-input performance. Distribution shift is the primary unresolved challenge.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 20px;">
<strong>The benchmark result is strong on clean images. Multi-seed evaluation is essential to draw reliable conclusions from small accuracy differences. Extending robustness to degraded inputs is the most important next step.</strong>
</div>

<!--
Compact CNNs work well on GTSRB under controlled conditions. Depth was the most effective architectural change. The LeakyReLU reversal demonstrates that multi-seed evaluation is not optional when differences are small. The noise result defines what this architecture still cannot do, and points directly to where future work should focus.
-->

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

## Questions?

<!--
Thank you. Backup slides are available on the autoencoder extension, hyperparameter sensitivity analysis, full per-run multi-seed table, and t-SNE feature space visualisation.
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
Optuna uses earlier trial results to guide subsequent ones. The key finding: our manual defaults were not fragile. The optimal learning rate is very close to our choice. The one notable gap is dropout — lower regularisation appears preferable for the Stride CNN architecture.
-->

---

## Backup: Multi-Seed Per-Run Results

<div class="kicker">All 15 training runs — seeds 42, 123, 2026</div>

| Seed | Model | Test Acc | Test Loss | Train Time | Epochs |
|---:|---|---:|---:|---:|---:|
| 42 | Baseline CNN | 99.20% | 0.0323 | 225 s | 16 |
| 42 | Deep CNN | 99.78% | 0.0077 | 293 s | 20 |
| 42 | LeakyReLU CNN | 99.63% | 0.0105 | 505 s | 20 |
| 42 | MobileNetV2 | 99.44% | 0.0197 | 530 s | 20 |
| 42 | Stride CNN | 99.46% | 0.0178 | 295 s | 20 |
| 123 | Baseline CNN | 99.64% | 0.0104 | 276 s | 20 |
| 123 | Deep CNN | 99.83% | 0.0075 | 289 s | 20 |
| 123 | LeakyReLU CNN | 99.69% | 0.0114 | 642 s | 20 |
| 123 | MobileNetV2 | 99.66% | 0.0105 | 529 s | 20 |
| 123 | Stride CNN | 99.30% | 0.0253 | 208 s | 14 |
| 2026 | Baseline CNN | 99.69% | 0.0118 | 281 s | 20 |
| 2026 | Deep CNN | 99.46% | 0.0190 | 216 s | 15 |
| 2026 | LeakyReLU CNN | 99.68% | 0.0124 | 645 s | 20 |
| 2026 | MobileNetV2 | 99.20% | 0.0299 | 529 s | 20 |
| 2026 | Stride CNN | 99.59% | 0.0120 | 300 s | 20 |

<!--
Use this slide if the professor asks for the raw per-run data behind the aggregated multi-seed table in the main deck. Seed 2026 for Deep CNN shows one example where Baseline outperformed Deep CNN — this is why the 0.18 pp mean advantage should be read as an average, not a guarantee.
-->

---

## Backup: Full Limitations

| Limitation | Impact |
|---|---|
| Internal hold-out split only | Not directly comparable to official GTSRB leaderboard |
| Pre-cropped images | Cannot locate signs — classification only, not detection |
| Three-seed validation, no independent splits | Relative ranking confirmed on average; cross-validation not performed |
| 32×32 input resolution | Fine details (numerals, symbols) may be partially lost |
| No noise or blur augmentation during training | Model not robust to degraded inputs |
| GTSRB: German roads, single camera, controlled weather | Limited generalisation to other regions, conditions, or sign standards |
| Rare classes have few test images | A single missed prediction shifts per-class accuracy significantly |

<!--
Use this slide for detailed questions about methodology. The most actionable limitations for follow-up work are noise augmentation, independent split validation, and higher resolution.
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
| 6 | Divider — Our Approach | 0:10 |
| 7 | Experimental design | 0:45 |
| 8 | Baseline architecture | 0:50 |
| 9 | Four variants | 1:00 |
| 10 | Training conditions | 0:40 |
| 11 | Divider — Results | 0:10 |
| 12 | Single-run results table | 0:40 |
| 13 | Main finding: Deep CNN | 0:50 |
| 14 | Multi-seed stability | 1:05 |
| 15 | Divider — Error Analysis | 0:10 |
| 16 | Error pattern | 0:45 |
| 17 | Confusion matrix (visual) | 0:25 |
| 18 | Bias analysis | 0:45 |
| 19 | Grad-CAM (visual) | 0:25 |
| 20 | Divider — Robustness | 0:10 |
| 21 | Robustness results | 0:50 |
| 22 | Limitations and next steps | 0:50 |
| 23 | Divider — Conclusion | 0:10 |
| 24 | Summary | 0:55 |
| 25 | Thank You | — |
| **Total** | | **~14:40** |
