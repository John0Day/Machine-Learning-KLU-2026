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
  .table-wrap {
    width: 92%;
    margin: 22px auto 20px auto;
  }
  .table-wrap table {
    margin-left: auto;
    margin-right: auto;
  }
  .compact-table table {
    font-size: 0.76em;
  }
  .slide-spacer {
    height: 10px;
  }
  .takeaway.tight {
    margin-top: 22px;
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
  section.centered {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .dataset-logos-badge {
    position: absolute;
    right: 48px;
    bottom: 48px;
    background: #ffffff;
    border-radius: 12px;
    padding: 12px 16px;
  }
  .dataset-logos-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .dataset-logos-row img {
    height: 40px !important;
    width: auto !important;
    max-width: none !important;
    display: block;
  }
---

<!-- _class: title -->
<!-- _paginate: false -->

<div class="kicker">Machine Learning Project · KLU 2026 · Shayan Razi & John Schlotfeldt</div>

# Compact CNN Architectures for Traffic Sign Classification

## Architecture, stability, and robustness on GTSRB

<p style="margin-top: 18px; font-size: 0.9em;">
German Traffic Sign Recognition Benchmark
</p>
<!--
Welcome everyone. This project is about teaching a computer to read traffic signs. We trained and compared five different neural network designs on a well-known benchmark dataset. Today we will walk you through what we built, what worked, and what the limitations are. The presentation will take about 15 minutes.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# The Dataset: German Traffic Sign Images from Ruhr University Bochum

<div class="subtitle">43 traffic sign classes and more than 39,000 labelled images.</div>

<div class="dataset-logos-badge">
  <div class="dataset-logos-row">
    <img src="sources/Ruhr-Universität_Bochum_logo.svg" alt="Ruhr University Bochum Logo" />
    <img src="sources/KLU_logo.png" alt="KLU Logo" />
  </div>
</div>

<!--
Let's start with some context. The next few slides explain what the task is, which dataset we used, and what makes it genuinely challenging.
-->

---

## Traffic sign classification — a core task in driver assistance systems

<div class="kicker">Background</div>

<div class="two-col" style="margin-top: 24px;">
  <div>
    <p><strong>What is the task?</strong></p>
    <p>Given a cropped image of a traffic sign, assign it to one of a fixed set of categories speed limits, prohibitory signs, warnings, mandatory directions.</p>
  </div>
  <div>
    <p><strong>Why does it matter?</strong></p>
    <p>Reliable sign recognition is a building block for driver assistance and autonomous systems. A model that misreads a speed limit or stop sign in real traffic has direct safety implications.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 28px;">
<strong>This project:</strong> we compare five CNN architectures on GTSRB a standard benchmark for this task to understand which design choices actually matter and how stable the results are.
</div>

<!--
Traffic sign classification means: given a cropped image of a sign, tell me what category it belongs to — speed limit, stop sign, warning triangle, and so on. That sounds straightforward, but it is a critical building block for any driver assistance or autonomous driving system. A car that misreads a speed limit or ignores a stop sign has a direct safety problem. We used GTSRB — a standard academic benchmark for this task — and our goal was simple: which neural network design choices actually improve accuracy, and do those improvements hold up when you test them more rigorously?
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
  <div><span class="step-label">Scope</span> Images are pre-cropped to the sign boundary. This is therefore a pure classification task, not a detection task in full-scene images.</div>
  <div><span class="step-label">Reference</span> The reported human recognition rate on GTSRB is <strong>98.84%</strong>. We use this as a broad performance reference, not a direct comparison target.</div>
</div>

<!--
The dataset is GTSRB — the German Traffic Sign Recognition Benchmark — with 39,209 images across 43 sign categories, all captured from a car-mounted camera on real German roads. A key detail: the images are already cropped to just the sign region, so the model only needs to classify, not locate. We split the data 70% for training, 15% for validation, and 15% for testing — giving us 5,881 test images. All images are resized to 32 by 32 pixels for a consistent input size. For reference, the human recognition rate on this dataset has been reported at 98.84% — we use this as a rough ballpark, not a direct comparison, because we are working with an internal split rather than the official test set.
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
Three things make this task harder than it looks. First, class imbalance. The most common sign — Speed limit 50 — has over 2,000 training examples. The rarest classes have around 147 each. That is a 10-to-1 ratio, which means the model sees some sign types very rarely during training. Second, visual ambiguity. Many signs share the same basic shape and differ only in a small number or symbol inside. A 30 and an 80 speed limit sign look almost identical at 32 by 32 pixels — the difference is just a handful of pixels. Third, within-class variation. Because the images were taken from a moving car, the same sign can look very different depending on the lighting, angle, or how blurry the image is. The model has to generalise across all of these conditions.
-->

---

## 43 classes — many share the same shape and differ only in a small internal symbol or numeral

<div style="display: flex; justify-content: center; margin: 8px 0;">

![w:680](results/task03/sample_images_by_class.png)

</div>

**Class IDs 0–42, left to right. Speed limit signs (classes 0–8) share the same red circular frame — the only distinguishing feature is the numeral inside.**

<div class="takeaway">
<strong>Key challenge:</strong> at 32×32 pixels, "30" and "80" differ by only a handful of pixels. The model must classify based on fine detail that is already near the resolution limit — without any additional context from the surrounding scene.
</div>

<!--
This grid shows one image per class. Look at the top row — those are all speed limit signs. They all share the same red circular frame. The only difference is the number inside. At this resolution, 30 and 80 are only a few pixels apart. This is not noise or a data quality issue — it is a structural property of how traffic signs are designed. Signs are standardised on purpose, because that makes them fast for humans to read on the road. But for a model working at 32 by 32 pixels, that standardisation becomes a challenge.
-->

---

## A 10.7× class imbalance built into the dataset

<div class="kicker">Class distribution</div>

![](results/task02/class_distribution.png)

<div class="takeaway">
  <strong>Speed limit 50:</strong> 2,250 images. <strong>Three rarest classes:</strong> 210 images each — a <strong>10.7× ratio</strong>. The 70/15/15 split inherits these proportions: the rarest classes train on only ≈147 images each.
</div>

<!--
This chart shows the raw class distribution. Speed limit 50, the tallest bar on the left, has over 2,000 images. The rare classes on the right have around 210 each. This reflects how often these signs actually appear in real German traffic — it is not a data collection error. Our 70-15-15 split inherits these proportions. To help the model learn from rare classes, we applied data augmentation — random rotations, brightness changes, and small shifts — during training. Whether this was enough, we tested later. The short answer is yes, mostly: the accuracy gap between frequent and rare classes turns out to be only 0.34 percentage points.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Our Approach

<div class="subtitle">One strong baseline. Four targeted architectural hypotheses.</div>

<!--
Now let's look at how we approached this. We started with one baseline architecture, established what it can do, and then tested four targeted variants — each one changing exactly one design decision.
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
Our baseline is a standard convolutional neural network with three processing stages. In stage one, 32 filters detect basic features like edges and colour boundaries. In stage two, 64 filters combine these into structural patterns like curves and sign outlines. In stage three, 128 filters assemble these into sign-level representations — things like a numeral inside a red circle. After these three stages, we have a 2,048-dimensional feature vector, which a final layer maps to one of the 43 sign classes. The whole network has 629,000 parameters and trains in roughly four to five minutes.
-->

---

## The baseline already reaches 99.49% — so where is the room to improve?


<div class="kicker">Baseline result — seed 42</div>

<div class="kpi-row">

  <div class="kpi"><div class="number">99.49%</div><div class="label">test accuracy</div></div>

  <div class="kpi"><div class="number">30</div><div class="label">wrong / 5,881</div></div>

  <div class="kpi"><div class="number">629K</div><div class="label">parameters</div></div>

</div>

<div class="takeaway">

<strong>Interpretation:</strong> the compact baseline already performs extremely well on clean, pre-cropped GTSRB images. The remaining errors motivate targeted architectural variants: more depth, transfer learning, alternative activation, and learned downsampling.

</div>

<!--
The baseline already reaches 99.49% on our test set — 30 wrong predictions out of 5,881. That is already a strong result for a compact model trained from scratch. But 30 errors remain, and when we looked at which images were wrong, they consistently involved visually similar class pairs — the same problem we saw in the data. That gives us a clear motivation for testing whether specific architectural changes can reduce those errors further.
-->
---

## The baseline leaves four architectural questions

<div class="kicker">From baseline to variants</div>

<div class="lead">The baseline is already strong, so each variant tests one plausible explanation for the remaining errors.</div>

<div class="cards-2">
  <div class="card">
    <h3>Depth</h3>
    <p>Are three feature extraction stages enough to separate fine numerals and small symbols?</p>
  </div>
  <div class="card">
    <h3>Transfer learning</h3>
    <p>Does pretrained visual knowledge help in a narrow but visually structured domain?</p>
  </div>
  <div class="card">
    <h3>Activation</h3>
    <p>Does the activation function affect training stability across different seeds?</p>
  </div>
  <div class="card">
    <h3>Downsampling</h3>
    <p>Does fixed pooling discard spatial detail that matters for small symbols?</p>
  </div>
</div>

<div class="takeaway">
<strong>Design logic:</strong> the variants are not chosen randomly. Each one targets a specific architectural question raised by the baseline result.
</div>

---

## Each variant maps one hypothesis to one model change

<div class="kicker">Controlled architectural comparison</div>

| Hypothesis | Variant | Main change |
|---|---|---|
| 3 stages may not resolve fine details at 32×32 | **Deep CNN** | Adds one extra feature extraction stage |
| Pretrained features may transfer to traffic signs | **MobileNetV2** | Uses an ImageNet-pretrained model |
| Activation choice may affect training stability | **LeakyReLU CNN** | Keeps small gradients for negative activations |
| Fixed pooling may lose relevant spatial detail | **Stride CNN** | Replaces fixed pooling with learned downsampling |

<div class="takeaway">
<strong>Controlled comparison:</strong> optimiser, split, augmentation, and evaluation protocol are identical across all five models — differences in accuracy are interpreted as architecture-driven.
</div>

<!--
Here is how each question maps to a concrete model. The Deep CNN adds one extra stage with 256 filters — testing whether more depth helps with fine visual details. MobileNetV2 replaces our custom network with one pretrained on 1.2 million images — testing whether prior visual knowledge transfers to traffic signs. LeakyReLU CNN swaps the activation function so that negative inputs still produce a small gradient instead of zero. And the Stride CNN replaces fixed max-pooling with a learnable downsampling step. Everything else — the optimizer, the data split, the augmentation, the number of epochs — is identical across all five models. That is what makes the comparison meaningful.
-->

---

## Identical training conditions make the architectural comparison as fair as possible

<div class="kicker">Training setup — fixed across all five models</div>

<div class="two-col">
  <div>
    <p><strong>Augmentation: training set only</strong></p>
    <ul>
      <li>Rotation ±15° simulates tilted camera angle</li>
      <li>Brightness and contrast variation</li>
      <li>Small translation shifts</li>
      <li>Validation and test: resize and normalise only, with no random transforms</li>
    </ul>
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

<div class="takeaway">
Any accuracy difference between models is attributable to architecture, not to differences in training conditions, data, or evaluation protocol.
</div>

<!--
A quick note on training setup. All five models receive the same augmentation during training — small rotations, brightness changes, and translation shifts. Validation and test sets get no random transforms, so every evaluation is reproducible. Same optimizer, same data split, same stopping criterion for everyone. This is what makes any accuracy difference we see attributable to the architecture, not to different training conditions.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Model Comparison

<div class="subtitle">All models exceed 99% on the internal split. Multi-seed analysis revised two of the initial rankings.</div>

<!--
Now let's look at the results. We will start with the single-run comparison, then show what happened when we tested each model three times with different random seeds.
-->

---

## Single-run results point to Deep CNN — but this is only the first signal

<div class="kicker">Canonical comparison — one training run, seed 42</div>

<div class="table-wrap compact-table">

| Model | Test Accuracy | Wrong / 5,881 | Parameters | Training Time |
|---|---:|---:|---:|---:|
| Baseline CNN | 99.49% | 30 | 629K | 275.6 s |
| **Deep CNN** | **99.81%** | **11** | **936K** | **284.0 s** |
| MobileNetV2 | 99.66% | 20 | 2.56M | 518.7 s |
| LeakyReLU CNN | 99.46% | 32 | 629K | 271.5 s |
| Stride CNN | 99.52% | 28 | 823K | 236.9 s |

</div>

<div class="takeaway tight">
<strong>First signal:</strong> Deep CNN has the strongest single-run result. But all models are above 99%, so small margins correspond to only a few images and need multi-seed validation.
</div>

<!--
Every model exceeds 99% accuracy. That tells us CNNs are well-suited for this task even at small scale. Deep CNN leads with 99.81%, followed by MobileNetV2 at 99.66%. LeakyReLU CNN is last at 99.46%. But here is the important thing: all the differences are within a fraction of a percent. At this scale, 0.1 percentage points is roughly 6 images. Differences that small can easily come from random initialization rather than from a genuine architectural advantage. So we cannot trust a single run. We need to test stability.
-->

---

<!-- _class: centered -->

## Additional depth produced the strongest single-run improvement

<div class="kicker">Deep CNN — single-run result (seed 42)</div>

<div class="kpi-row" style="margin-top: 30px; margin-bottom: 18px;">
  <div class="kpi"><div class="number">99.81%</div><div class="label">test accuracy</div></div>
  <div class="kpi"><div class="number">11</div><div class="label">wrong / 5,881</div></div>
  <div class="kpi"><div class="number">+8 s</div><div class="label">training time vs. baseline</div></div>
</div>

<div class="takeaway" style="margin-top: 34px;">
<strong>Single-run signal:</strong> adding one feature extraction stage reduced errors from <strong>30 to 11</strong>, a <strong>63% reduction</strong>, with only moderate parameter growth. The next slide checks whether this result remains stable across seeds.
</div>

<!--
To make the Deep CNN result concrete: adding one extra convolutional stage reduced errors from 30 to 11 — a 63% reduction — while training time went up by only 8 seconds. That is a very favourable cost-benefit ratio. But this is still a single run. The next slide shows whether the result holds across multiple seeds.
-->

---

## Multi-seed validation gives the more reliable ranking

<div class="kicker">3 seeds × 5 models = 15 training runs</div>

<div class="table-wrap compact-table">

| Rank | Model | Test Acc (mean ± std) | Parameters | Train Time |
|---:|---|---:|---:|---:|
| 1 | **Deep CNN** | **99.69% ± 0.17%** | 936K | 266 s |
| 2 | **LeakyReLU CNN** | **99.67% ± 0.03%** | 629K | 597 s |
| 3 | Baseline CNN | 99.51% ± 0.22% | 629K | 261 s |
| 4 | Stride CNN | 99.45% ± 0.12% | 823K | 268 s |
| 5 | MobileNetV2 | 99.43% ± 0.19% | 2.56M | 529 s |

</div>

<div class="takeaway tight">
<strong>Key message:</strong> Deep CNN remains strongest on average. LeakyReLU is almost as accurate and very stable, but slower. MobileNetV2 shows no stable advantage despite higher cost.
</div>

<!--
We trained all five models three times each with different random seeds — 15 runs in total. Two models changed their story significantly. LeakyReLU CNN was last in the single run at 99.46%. Across three seeds, its mean is 99.67% — second place — with the smallest variance of all models at ±0.03%. The seed-42 result was simply an outlier. This is a strong argument for never relying on a single training run. MobileNetV2 went the other direction: it looked third-best in the single run at 99.66%, but its multi-seed mean drops to last place at 99.43% — despite having substantially more parameters and nearly twice the training time of the baseline. Deep CNN remains first on average.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Deep CNN Evaluation

<div class="subtitle">Selected for detailed analysis: strongest average accuracy with moderate parameter growth and near-baseline training cost.</div>

<!--
Deep CNN has the best average result with reasonable parameter growth and near-baseline training cost, so we selected it for a more detailed analysis. The next slides look at where the errors come from, what the confusion matrix shows, whether the class imbalance hurt rare classes, and what the model is actually looking at when it makes a prediction.
-->

---

## Error profile — all 11 errors involve visually similar class pairs

<div class="kicker">Deep CNN — test set (seed 42, 5,881 images)</div>

<div style="margin: 20px 0 18px 0;">

| Class | Per-class accuracy | Why it is hard |
|---|:---:|---|
| Pedestrians (class 27) | 97.62% | Near-identical triangular silhouette to Bicycles crossing |
| Bicycles crossing (class 29) | 97.62% | Same shape as Pedestrians at 32×32 resolution |
| Speed limit 120 km/h (class 8) | 99.10% | "120" vs. "100" differ by only a few pixels |

</div>

<div class="takeaway">
<strong>Pattern:</strong> every error occurs at a class boundary where the distinguishing feature — a numeral or small symbol — occupies only a handful of pixels.
</div>

<!--
With only 11 errors, we could look at each one individually. The pattern is very clear: every single error involves two sign classes that look nearly identical at 32 by 32 pixels. Pedestrians and Bicycles crossing share almost the same triangular silhouette. Speed limit 120 and Speed limit 100 differ by one digit. The model's Top-5 accuracy is 99.98% — meaning the correct class is in the model's top five predictions in all but one of the 5,881 test cases. This tells us the model is not confused — it is working at the limit of the resolution. Higher resolution images would likely fix most of these errors.
-->

---

<!-- _class: image-focus -->

## Misclassified examples — all involve visually similar class pairs

![](results/task06/deep/misclassifications_top_confidence.png)

**Representative examples of the 11 errors: mostly speed limits differing by one digit and warning triangles with very similar silhouettes.**

<!--
These are the actual misclassified images from the test set. You can see why they are hard — at this resolution, signs from different classes look almost identical. The model is not making random mistakes. It is failing at exactly the hardest cases in the dataset, where the distinguishing feature is only a few pixels wide.
-->

---

<!-- _class: image-focus -->

## The confusion matrix shows near-perfect diagonal with minimal off-diagonal activity

![](results/task06/deep/confusion_matrix_normalized.png)

**Top-5 accuracy: 99.98% — the correct class appeared in the model's top 5 predictions in all but one of 5,881 test cases.**

<!--
The confusion matrix shows all 43 classes. A perfect model would have only values on the diagonal. Here, the diagonal is almost entirely filled, confirming that the model classifies correctly for the vast majority of classes. The few off-diagonal entries you can see cluster around the same visually similar pairs we just discussed. The Top-5 accuracy of 99.98% means the correct class appears in the top five predictions in all but one case.
-->

---

## Frequency bias analysis — the 10.7× imbalance produced only a 0.34 pp accuracy gap

<div class="kicker">Frequency bias — Deep CNN</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.87%</div><div class="label">mean accuracy — 10 most frequent classes</div></div>
  <div class="kpi"><div class="number">99.52%</div><div class="label">mean accuracy — 10 rarest classes</div></div>
  <div class="kpi"><div class="number">0.34 pp</div><div class="label">gap</div></div>
</div>

<div class="takeaway" style="margin-top: 24px;">
<strong>Interpretation:</strong> despite a 10.7× data imbalance, the accuracy gap between frequent and rare classes is only 0.34 pp. Several rare classes even reach <strong>100%</strong> — augmentation appears to have partially compensated for the data shortage. This result is indicative; rare classes have too few test images for a conclusive estimate.
</div>

<!--
Remember the 10-to-1 class imbalance we saw at the beginning? We tested whether it actually hurt performance on rare classes. The answer is: not much. The accuracy gap between the 10 most frequent classes and the 10 rarest is only 0.34 percentage points — 99.87% versus 99.52%. Some rare classes even reach 100% accuracy. The augmentation strategy appears to have partially compensated for the limited training data. One important caveat though: rare classes have very few test images, so a single missed prediction shifts the per-class accuracy significantly. This result is a positive signal, but it is indicative rather than conclusive.
-->

---

<!-- _class: image-focus -->

## Grad-CAM indicates that predictions are driven by the sign region, not background context

![](results/task06/deep/gradcam_examples.png)

**Activation maps concentrate on the sign shape and internal symbol — consistent with task-relevant feature learning.**

<!--
Finally, we used a technique called Grad-CAM to visualise where the model is looking when it makes a prediction. The activation maps show which parts of the image had the most influence. In almost every case, the activations concentrate on the sign itself — specifically on the shape and the symbol or number inside — rather than on the surrounding road or background. This is reassuring: the model is learning to use the right information, not background shortcuts.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Robustness

<div class="subtitle">The Deep CNN performs well under controlled conditions — but real traffic images are not always clean.</div>

<!--
Everything we have seen so far was measured on clean, well-cropped benchmark images. But real traffic images are not always clean. Let's test what happens when the input quality is degraded.
-->

---

## Gaussian noise causes a 27.95 pp accuracy drop, the main robustness limitation

<div class="kicker">Robustness test — no retraining</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.81%</div><div class="label">clean test images</div></div>
  <div class="kpi"><div class="number">97.01%</div><div class="label">Gaussian blur −2.80 pp</div></div>
  <div class="kpi"><div class="number" style="color: #c0392b;">71.86%</div><div class="label">Gaussian noise −27.95 pp</div></div>
</div>

<br>

The model handles moderate blur reasonably well. Under Gaussian noise, accuracy drops from **99.81%** to **71.86%** roughly **1,600 additional errors** on the same 5,881 test images.

<div class="takeaway">
<strong>Cause:</strong> this is a distribution shift failure. The model was trained on clean images and was not exposed to noisy inputs during training. The clean benchmark result therefore does not fully describe performance under degraded real-world conditions.
</div>

<!--
We applied two types of degradation to the test set without any retraining. Gaussian blur — simulating motion blur or an out-of-focus camera — caused a drop of only 2.8 percentage points, from 99.81% to 97.01%. That is actually encouraging. The model has learned features that do not completely depend on sharp edges. But Gaussian noise — simulating a poor sensor or a low-quality camera — caused a drop of 27.95 percentage points, down to 71.86%. That is roughly 1,600 additional wrong predictions on the same test set. The cause is simple: the model was never trained on noisy images, so it has no learned strategy for handling them. Clean benchmark accuracy and noisy-input accuracy are two very different things.
-->

---

## This work is a controlled first step — three directions for improvement

<div class="kicker">Limitations and next steps</div>

<div class="two-col">
  <div>
    <p><strong>Constraints to keep in mind</strong></p>
    <ul>
      <li>Pre-cropped images: the model classifies signs but does not locate them</li>
      <li>Image sizes larger than 32x32 pixels require more computational resources</li>
      <li>Clean benchmark conditions: robustness to noise and weather remains limited</li>
      <li>German roads only: generalisation to other sign systems is untested</li>
    </ul>
  </div>
  <div>
    <p><strong>Next steps</strong></p>
    <ul>
      <li><strong>Short term:</strong> add noise and blur augmentation to the baseline model to address the robustness gap</li>
      <li><strong>Medium term:</strong> validate across independent data splits to confirm ranking stability</li>
      <li><strong>Longer term (object detection):</strong> move from classifying pre-cropped signs to finding and recognising them in a live camera stream</li>
    </ul>
  </div>
</div>

<div class="takeaway">
This classifier is a building block. The path to a deployable system requires robustness to real conditions and the ability to detect signs before classifying them.
</div>

<!--
To summarise the limitations: the model classifies pre-cropped signs — it cannot find a sign in a full scene. It was only tested on German roads under controlled conditions. And robustness to degraded input is clearly limited. For next steps: in the short term, the most direct fix is adding noise and blur augmentation during training — this addresses the robustness gap without needing a new architecture. In the medium term, testing across independent data splits would strengthen our confidence in the model ranking. And in the longer term, the natural direction is object detection: building a system that can find the sign in a live camera frame and classify it, rather than starting from an already-cropped image.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Conclusion

<div class="subtitle">Depth was strongest. LeakyReLU surprised in multi-seed evaluation. Robustness remains open.</div>

<!--
Three findings to close with. I will keep this brief.
-->

---

## Three findings from this experiment

<div class="kicker">Summary</div>

<div class="cards">
  <div class="card">
    <h3>Compact CNNs are sufficient here</h3>
    <p>A 629K-parameter baseline trained from scratch reaches <strong>99.49%</strong> on this internal split in the same broad performance range as the reported human benchmark of 98.84%.</p>
  </div>
  <div class="card">
    <h3>Architecture matters, but so does validation</h3>
    <p>Depth reduced errors by <strong>63%</strong> at near-zero cost, making it the strongest single change. Multi-seed analysis then showed LeakyReLU CNN as second-best overall, so the single-run last place was a misleading outlier.</p>
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
Three takeaways. First: compact CNNs are sufficient for this benchmark. Even our smallest model with 629,000 parameters reaches 99.49% accuracy — well above the reported human benchmark. Second: architecture matters, but so does how you evaluate. Adding depth was the strongest single change, cutting errors by 63%. But the multi-seed analysis showed that LeakyReLU CNN — which looked worst in the single run — was actually second-best overall. When accuracy differences are this small, a single training run is not enough to draw reliable conclusions. Third: robustness is the open problem. A 27-percentage-point drop under noise shows that strong benchmark performance does not mean the model is ready for real-world deployment. That is where the most important work remains.
-->

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

## Questions?

<!--
Thank you for listening. I am happy to take any questions. If there are questions about specific numbers or methodology, backup slides are available with the full per-run multi-seed data, the complete limitations table, hyperparameter sensitivity analysis, and an anomaly detection extension using an autoencoder.
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

![](results/task05/tsne_feature_space.png)

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
