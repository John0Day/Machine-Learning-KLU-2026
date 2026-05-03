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
  .cards-5 {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-top: 18px;
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
  section::before {
    content: '';
    position: absolute;
    top: 14px;
    right: 22px;
    width: 600px;
    height: 100px;
    background-image: url('sources/KLU_logo.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: right center;
    z-index: 10;
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

<div class="kicker">Deep Learning Project · KLU 2026 · Shayan Razi & John Schlotfeldt</div>

# Deep Learning for Traffic Sign Classification

## Building and evaluating a CNN-based classifier on GTSRB

<p style="margin-top: 18px; font-size: 0.9em;">
German Traffic Sign Recognition Benchmark · 43 classes · deep learning evaluation
</p>
<!--
Good morning everyone. This project is about applying deep learning to traffic sign classification on the GTSRB benchmark. I will first introduce the dataset and the classification task, then explain how we build a CNN-based baseline and evaluate its performance. From there, the presentation gradually moves toward model design choices, evaluation stability, and the limitations we found under degraded inputs.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Background & Dataset

<div class="subtitle">What is the task, why does it matter, and what makes GTSRB a challenging benchmark.</div>

<div class="dataset-logos-badge">
  <div class="dataset-logos-row">
    <img src="sources/Ruhr-Universität_Bochum_logo.svg" alt="Ruhr University Bochum Logo" />
    <img src="sources/KLU_logo.png" alt="KLU Logo" />
  </div>
</div>

<!--
The dataset we used is GTSRB (the German Traffic Sign Recognition Benchmark), collected by Ruhr University Bochum. One important thing to note upfront: all images are already cropped to the sign boundary, so we are solving a pure classification problem, not a detection problem.
-->

---

## Traffic sign classification: a core task in driver assistance systems

<div class="kicker">Background</div>

<div class="two-col" style="margin-top: 24px;">
  <div>
    <p><strong>What is the task?</strong></p>
    <p>Given a cropped image of a traffic sign, assign it to one of a fixed set of categories: speed limits, prohibitory signs, warnings, mandatory directions.</p>
  </div>
  <div>
    <p><strong>Why does it matter?</strong></p>
    <p>Reliable sign recognition is a building block for driver assistance and autonomous systems. A model that misreads a speed limit or stop sign in real traffic has direct safety implications.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 28px;">
<strong>This project:</strong> we compare five CNN architectures on GTSRB, a standard benchmark for this task, to understand which design choices actually matter and how stable the results are.
</div>

<!--
We start with the basic task: the model receives a cropped image of a traffic sign and has to assign it to one of 43 categories. This matters because traffic signs communicate safety-relevant instructions, such as speed limits, warnings, and mandatory directions. In a driver assistance context, misreading one of these signs could lead to a wrong driving decision. That is why GTSRB is a useful benchmark for this project: it lets us compare classification approaches on a controlled, safety-relevant task.
-->

---

## GTSRB: a real-world classification benchmark with structural challenges

<div class="kicker">Dataset</div>

<div class="lead">39,209 labelled images across 43 classes, captured from a car-mounted camera on real German roads.</div>

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
The dataset has 39,209 images across 43 classes, all captured from a moving vehicle on real German roads. The reported human recognition rate of 98.84 percent gives useful context for interpreting our results, but since we use our own internal split rather than the official test set, it is a reference point, not a direct comparison target.
-->

---

## Three dataset properties shape the evaluation strategy

<div class="kicker">Problem structure</div>

<div class="cards">
  <div class="card">
    <h3>Class imbalance</h3>
    <p>The most frequent class has <strong>2,250</strong> images. The rarest have <strong>210</strong>. After our 70/15/15 split, the rarest classes train on roughly <strong>147 examples</strong> each.</p>
  </div>
  <div class="card">
    <h3>Visual ambiguity</h3>
    <p>Many classes share shape and colour. This makes class-level error analysis necessary, especially for signs with small internal details.</p>
  </div>
  <div class="card">
    <h3>Within-class variation</h3>
    <p>Lighting, blur, contrast, and camera angle vary within each class. The model must generalise across these conditions from limited training data.</p>
  </div>
</div>

<div class="takeaway">
<strong>Implication:</strong> overall accuracy alone is not enough. Class imbalance motivates frequency-bias analysis, visual ambiguity motivates error analysis, and input variation motivates robustness testing.
</div>

<!--
Before looking at individual examples, we should separate the three dataset properties that shape the rest of our evaluation. First, class imbalance means that high overall accuracy could still hide weaker performance on rare traffic signs, so we later compare frequent and rare classes explicitly. Second, visual ambiguity means that some classes share shape and colour and differ only in small internal details, which is why we later look at error patterns and misclassified examples. Third, within-class variation means that the same sign can appear under different lighting, blur, contrast, and viewing angles, which motivates the robustness tests later in the presentation. So the key message is: overall accuracy alone is not enough for evaluating this model properly.
-->

---

## Zoom-in: many signs differ only in small internal details

![bg right:48% contain](results/task03/sample_images_by_class.png)

<div style="width: 47%; margin-top: 26px;">

<div class="kicker">Visual ambiguity · original dataset images</div>

<div class="lead" style="font-size: 1.22em; line-height: 1.28; margin-top: 10px;">
Many signs share the same outer shape, so the decisive information is often a small numeral or symbol.
</div>

<div class="takeaway" style="margin-top: 26px; font-size: 0.95em;">
<strong>Key challenge:</strong> signs such as speed limit 30 and 80 share the same red circular frame — the only difference is the numeral inside. The model must learn to distinguish classes based on fine internal detail.
</div>

</div>

<!--
After the general dataset challenges, we zoom in on the visual ambiguity problem. Speed limit signs are a good example: they share the same red circular frame, and the only difference is the numeral inside. The model has to separate classes based on that fine internal detail alone — there is no shape or colour difference to rely on. This is why visually similar classes show up again later in the error analysis.
-->

---

## Class imbalance motivates the frequency-bias analysis

<div class="kicker">Class distribution</div>

![](results/task02/class_distribution.png)

<div class="takeaway">
<strong>Key implication:</strong> the rarest classes have only 210 images before splitting and roughly 147 training examples after the 70/15/15 split. This is why we later check whether rare classes perform worse than frequent ones.
</div>

<!--
This plot makes the class imbalance visible. Some classes, such as Speed limit 50, have thousands of examples, while the rarest classes have only 210 images before the split. After the 70, 15, 15 split, those rare classes contribute only about 147 training images. This matters because high overall accuracy could still hide weaker performance on rare signs, which is why we later include a frequency-bias analysis.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Our Approach

<div class="subtitle">From preprocessing to a compact CNN baseline under controlled conditions.</div>

<!--
Now that the dataset challenges are clear, we can move to the modelling approach. I will first show how the raw images are turned into fixed model inputs, because preprocessing defines what every model actually sees. After that, I introduce the compact CNN baseline as the reference model for the rest of the experiment.
-->

---

## Preprocessing pipeline: from raw image to model input

<div class="kicker">Training only · augmentation simulates real-world variation</div>

<div class="cards-5">
  <div class="card" style="text-align:center;">
    <span class="step-label">1</span>
    <h3>Resize</h3>
    <p style="font-size:0.85em;">32×32 px<br><span class="muted">originals: 25–243 px</span></p>
  </div>
  <div class="card" style="text-align:center;">
    <span class="step-label">2</span>
    <h3>Rotation</h3>
    <p style="font-size:0.85em;">±15°<br><span class="muted">tilted camera angles</span></p>
  </div>
  <div class="card" style="text-align:center;">
    <span class="step-label">3</span>
    <h3>ColorJitter</h3>
    <p style="font-size:0.85em;">brightness, contrast, saturation<br><span class="muted">lighting &amp; weather</span></p>
  </div>
  <div class="card" style="text-align:center;">
    <span class="step-label">4</span>
    <h3>Translate</h3>
    <p style="font-size:0.85em;">±10% shift<br><span class="muted">off-centre sign</span></p>
  </div>
  <div class="card" style="text-align:center;">
    <span class="step-label">5</span>
    <h3>Normalize</h3>
    <p style="font-size:0.85em;">GTSRB mean &amp; std<br><span class="muted">zero mean, unit variance</span></p>
  </div>
</div>

<div class="cards-2" style="margin-top: 14px;">
  <div class="takeaway" style="margin-top:0;">
    <strong>Val / Test:</strong> Resize + Normalize only — no random transforms, identical input every run.
  </div>
  <div class="takeaway" style="margin-top:0;">
    <strong>Split (seed 42):</strong> 70 / 15 / 15 → 27,447 train · 5,881 val · 5,881 test images.
  </div>
</div>

<!--
Before any model sees the data, every image goes through the same preprocessing pipeline. The resize to 32 by 32 is our choice, not a dataset property — originals range from 25 to 243 pixels. Augmentation is applied only to training images to simulate real-world conditions: slight rotation for camera angle, color jitter for lighting and weather, and a small translation for off-centre signs. Validation and test images receive only resize and normalise, with no randomness, so evaluation results are reproducible across runs.
-->

---

<!-- _class: image-focus -->

## What augmented training images look like

<div class="kicker">16 training samples after augmentation · 32×32 px</div>

<div class="two-col" style="grid-template-columns: 62% 38%; gap: 26px; align-items: center; margin-top: 12px;">
  <div style="display: flex; justify-content: center; align-items: center;">
    <img src="results/task03/preprocessing_sample_grid.png" style="max-width: 100%; max-height: 56vh; object-fit: contain;" />
  </div>
  <div class="takeaway" style="margin-top: 0; font-size: 0.95em;">
    <strong>Training input:</strong> each image has already been resized, rotated, colour-jittered, and translated. This is the exact augmented input the model learns from.
  </div>
</div>

<!--
This grid shows 16 real training images after all augmentation steps have been applied. The effect is visible: images appear at slightly different angles, with varying brightness and contrast. This is what the model actually learns from — not the clean originals.
-->

---

## The baseline processes images in three hierarchical stages

<div class="kicker">Baseline architecture · 629K parameters</div>

<div class="lead">Each stage learns increasingly abstract representations from the input image.</div>

<div class="cards">
  <div class="card">
    <h3><span class="step-label">Stage 1</span> Low-level features</h3>
    <p>32 filters learn edges, colour boundaries, and brightness gradients. Spatial resolution is reduced via pooling.</p>
  </div>
  <div class="card">
    <h3><span class="step-label">Stage 2</span> Structural patterns</h3>
    <p>64 filters combine earlier features into corners, curves, and sign contours. Resolution is reduced again.</p>
  </div>
  <div class="card">
    <h3><span class="step-label">Stage 3</span> Sign representations</h3>
    <p>128 filters learn sign-level patterns such as numerals in circles and symbols in triangles. Output: a 2,048-value feature representation.</p>
  </div>
</div>

<div class="takeaway">
<strong>Classifier:</strong> after the three feature stages, the model compresses the image into a 2,048-value representation and maps it to one of 43 traffic sign classes. The full model is trained end to end on GTSRB.
</div>

<!--
The baseline model processes images in three hierarchical stages. The first stage learns basic visual patterns such as edges and colour boundaries, the second combines these into shapes and contours, and the third builds more sign-specific representations such as numerals inside circles. After these stages, the model compresses the image into a 2,048-value feature representation and uses it to predict one of the 43 classes. With 629,000 parameters and training from scratch on GTSRB, this is our reference point for everything that follows.
-->

---

## The baseline already reaches 99.49%. Where is the room to improve?

<div class="kicker">Baseline result · seed 42</div>

<div class="kpi-row">

  <div class="kpi"><div class="number">99.49%</div><div class="label">test accuracy</div></div>

  <div class="kpi"><div class="number">30</div><div class="label">wrong / 5,881</div></div>

  <div class="kpi"><div class="number">629K</div><div class="label">parameters</div></div>

</div>

<div class="takeaway">

<strong>Interpretation:</strong> the compact baseline already performs extremely well on clean, pre-cropped GTSRB images. The remaining errors make it meaningful to test whether specific architectural choices can improve this already strong reference point.

</div>

<!--
The baseline reaches 99.49 percent on the internal test split, with only 30 wrong predictions out of 5,881 images. It does this with 629K parameters, so it is already a strong and relatively compact reference model. That raises the central question for the next part: if the baseline is already this good, can specific architectural changes still improve it in a meaningful way? This is what motivates the variants on the next slide.
-->

---

## The baseline leaves four architectural questions

<div class="kicker">From baseline to variants</div>

<div class="lead">The baseline is already strong, so each variant tests one architectural hypothesis for improving or stabilising performance.</div>

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
    <p>Does the activation function affect training stability?</p>
  </div>
  <div class="card">
    <h3>Downsampling</h3>
    <p>Does fixed pooling discard spatial detail that matters for small symbols?</p>
  </div>
</div>

<div class="takeaway">
<strong>Design logic:</strong> the variants are not chosen randomly. Each one tests a specific architectural hypothesis motivated by the baseline result.
</div>

<!--
At this point, the baseline gives us a strong reference point, but it also raises the question of what could still be improved. We therefore define four architectural hypotheses rather than selecting variants randomly. More depth might help with fine visual details, transfer learning might provide useful visual features, the activation function might affect training stability, and learned downsampling might preserve spatial information better than fixed pooling. The next slide maps these four questions to the concrete model variants we tested.
-->

---

## Each variant maps one hypothesis to one model change

<div class="kicker">From questions to model variants</div>

| Hypothesis | Variant | Main change |
|---|---|---|
| 3 stages may not resolve fine details at 32×32 input size | **Deep CNN** | Adds one extra feature extraction stage |
| Pretrained features may transfer to traffic signs | **MobileNetV2** | Uses an ImageNet-pretrained model |
| Activation choice may affect training stability | **LeakyReLU CNN** | Keeps small gradients for negative activations |
| Fixed pooling may lose relevant spatial detail | **Stride CNN** | Replaces fixed pooling with learned downsampling |

<div class="takeaway">
<strong>Design logic:</strong> the four variants translate the previous architectural questions into concrete model changes. The next slide shows how we keep the training and evaluation setup fixed across all models.
</div>

<!--
The previous slide introduced the four architectural questions. Here, each question is mapped to the model variant that tests it. Deep CNN tests additional depth, MobileNetV2 tests transfer learning, LeakyReLU tests activation choice, and Stride CNN tests learned downsampling. These are not random alternatives, but targeted variants around one main design aspect. The next step is to make sure all of them are trained and evaluated under the same conditions.
-->

---

## Controlled training setup makes the model comparison interpretable

<div class="kicker">Training setup · fixed across all five models</div>

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
<strong>Controlled setup:</strong> because data split, augmentation, training settings, and evaluation protocol are fixed, performance differences can be interpreted primarily as model-design effects, while still considering parameter count and training time.
</div>

<!--
The comparison is only meaningful because the training and evaluation setup is fixed across all five models. Augmentation is applied only to the training set, while validation and test images are resized and normalised without random transforms. Early stopping monitors validation loss and halts training once it stops improving — this prevents overfitting and means models that converge faster simply finish earlier rather than continuing to train on noise. This means performance differences can be interpreted mainly through model design, while still considering parameter count, training time, and seed variation. The goal is a controlled architectural comparison rather than a tournament between unrelated models.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Model Comparison

<div class="subtitle">All models exceed 99% on the internal split. Multi-seed validation gives the more reliable ranking.</div>

<!--
With the setup clear, we can move to the model results. I first show the canonical single-run comparison under seed 42, because it gives an initial signal about which architecture looks strongest. Then I move to the multi-seed analysis across three seeds, which is the more reliable basis for drawing conclusions when all models are already above 99 percent.
-->

---

## Single-run results point to Deep CNN, but this is only the first signal

<div class="kicker">Canonical comparison · one training run, seed 42</div>

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
<strong>First signal:</strong> Deep CNN has the strongest single-run result, reducing errors from <strong>30 to 11</strong> compared with the baseline. But all models are above 99%, so small margins correspond to only a few images and need multi-seed validation.
</div>

<!--
Deep CNN comes out on top in this single run with 99.81 percent and only 11 wrong predictions out of 5,881 images. Compared with the baseline, that reduces the error count from 30 to 11, so depth looks like the strongest first signal. However, all five models are above 99 percent, and the gap between first and last is still less than 0.4 percentage points. At this level, a few images can change the ranking, which is why the next step is the multi-seed analysis.
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
Across three seeds, Deep CNN maintains the strongest average at 99.69 percent. The bigger story is LeakyReLU CNN: it looked weak in the single-run result, but averaged across seeds it is second-best and extremely stable, with a standard deviation of only 0.03 percent. MobileNetV2 drops to last place on average despite being more than four times larger than the baseline. The rankings you draw from a single run can be genuinely misleading. With Deep CNN confirmed as the strongest model on average, the next section takes a closer look at what its remaining errors actually look like.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Deep CNN Evaluation

<div class="subtitle">Selected for detailed analysis because it offers the strongest average accuracy-cost tradeoff.</div>

<!--
After the model comparison, we now focus on the selected Deep CNN. It has the strongest average accuracy-cost tradeoff, so it is the most relevant model to analyse in detail. The goal here is to understand not just how accurate it is, but where it still makes mistakes, whether those mistakes follow a pattern, and whether the model appears to focus on sign-relevant regions.
-->

---

## Error profile: remaining mistakes cluster around visually similar classes

<div class="kicker">Deep CNN · test set (seed 42, 5,881 images)</div>

<div style="margin: 20px 0 18px 0;">

| Class | Per-class accuracy | Why it is hard |
|---|:---:|---|
| Pedestrians (class 27) | 97.62% | Near-identical triangular silhouette to Bicycles crossing |
| Bicycles crossing (class 29) | 97.62% | Same shape as Pedestrians at our 32×32 input size |
| Speed limit 120 km/h (class 8) | 99.10% | "120" vs. "100" differ by only a few pixels |

</div>

<div class="takeaway">
<strong>Pattern:</strong> remaining errors cluster around class boundaries where the distinguishing feature, such as a numeral or small symbol, occupies only a handful of pixels.
</div>

<!--
The Deep CNN makes only 11 errors on the test set, so we can look at the remaining mistakes quite concretely. The weaker classes shown here are visually difficult cases: pedestrians and bicycles share a similar triangular layout, and speed limit 120 can be close to speed limit 100 after resizing. The pattern suggests that the model is not failing broadly, but mainly struggles where small visual details are close to the resolution limit.
-->

---

<!-- _class: image-focus -->

## Misclassified examples make the error pattern visible

![](results/task06/deep/misclassifications_top_confidence.png)

**Representative examples of the 11 errors: mostly speed limits differing by one digit and warning triangles with very similar silhouettes.**

<!--
Looking at the actual misclassified images makes the error pattern easier to understand. Most examples involve speed limit signs differing by a small numeral, or warning signs with very similar triangular silhouettes. These are exactly the kinds of cases where resizing to 32 by 32 can make the decisive details difficult to resolve. A natural follow-up would be to test whether a higher input resolution reduces these errors.
-->

---

## Errors are rare and concentrated, not randomly distributed

<div class="kicker">Confusion matrix · Deep CNN</div>

<div class="two-col" style="grid-template-columns: 58% 42%; gap: 24px; align-items: center; margin-top: 10px;">
  <div style="display: flex; justify-content: center; align-items: center;">
    <img src="results/task06/deep/confusion_matrix_normalized.png" style="max-width: 100%; max-height: 62vh; object-fit: contain;" />
  </div>
  <div>
    <div class="lead" style="font-size: 1.08em; line-height: 1.3; margin-bottom: 14px;">
      The strong diagonal shows that most classes are predicted correctly.
    </div>
    <div class="takeaway" style="font-size: 0.92em; margin-top: 0;">
      <strong>Interpretation:</strong> the few off-diagonal entries are sparse and align with visually similar signs, rather than being randomly spread across all 43 classes.
    </div>
    <p class="muted" style="font-size: 0.84em; margin-top: 16px;">
      Top-5 accuracy: <strong>99.98%</strong>. The correct class appeared in the model's top 5 predictions in all but one of 5,881 test cases.
    </p>
  </div>
</div>

<!--
After looking at individual examples, the confusion matrix gives the class-level view. The strong diagonal shows that errors are rare overall, and the few off-diagonal entries are not spread randomly across the class space. Instead, they mostly align with the visually similar sign groups we already discussed. The Top-5 accuracy of 99.98 percent also shows that even when the top prediction is wrong, the correct class is almost always still among the model's most likely alternatives.
-->

---

## Frequency bias appears limited in this run

<div class="kicker">Frequency bias · Deep CNN</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.87%</div><div class="label">mean accuracy, 10 most frequent classes</div></div>
  <div class="kpi"><div class="number">99.52%</div><div class="label">mean accuracy, 10 rarest classes</div></div>
  <div class="kpi"><div class="number">0.34 pp</div><div class="label">gap</div></div>
</div>

<div class="takeaway" style="margin-top: 24px;">
<strong>Interpretation:</strong> despite a 10.7× data imbalance, the accuracy gap between frequent and rare classes is only 0.34 pp in this run. This suggests no strong class-frequency bias under this setup. However, rare classes have few test images, so the result should be interpreted cautiously.
</div>

<!--
After looking at where the errors occur, we now ask a different question: is the model systematically worse on rare classes? One concern with any imbalanced dataset is that high overall accuracy could hide weak performance on underrepresented signs. In this run, the gap between the 10 most frequent and 10 rarest classes is only 0.34 percentage points, which suggests no strong class-frequency bias under this setup. However, rare classes have few test images, so this result should be interpreted cautiously.
-->

---

<!-- _class: image-focus -->

## Grad-CAM suggests that predictions rely on sign-relevant regions

![](results/task06/deep/gradcam_examples.png)

**Activation maps concentrate on the sign shape and internal symbol, consistent with task-relevant feature learning.**

<!--
After checking error concentration and class-frequency effects, Grad-CAM gives us a qualitative look at what image regions influence the model. The activation maps mostly concentrate on the sign shape and internal symbol, rather than on obvious background areas. This does not prove the full decision process, but it supports the interpretation that the model is using task-relevant visual information. That is useful because background shortcuts would be a concern for generalisation.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Robustness

<div class="subtitle">The Deep CNN performs well under controlled conditions, but real traffic images are not always clean.</div>

<!--
So far, the Deep CNN looked very strong on clean, pre-cropped benchmark images. The next question is whether that performance still holds when the input becomes less ideal. This matters because real camera images can contain blur, sensor noise, exposure changes, or weather effects. So here we deliberately degrade the test images and check how much performance drops without retraining the model.
-->

---

## Gaussian noise causes a 27.80 pp accuracy drop, the main robustness limitation

<div class="kicker">Robustness test · no retraining</div>

<div class="kpi-row">
  <div class="kpi"><div class="number">99.81%</div><div class="label">clean test images</div></div>
  <div class="kpi"><div class="number">97.01%</div><div class="label">Gaussian blur −2.80 pp</div></div>
  <div class="kpi"><div class="number" style="color: #c0392b;">72.01%</div><div class="label">Gaussian noise −27.80 pp</div></div>
</div>

<br>

The model handles moderate blur reasonably well. Under Gaussian noise, accuracy drops from **99.81%** to **72.01%**, corresponding to roughly **1,650 additional errors** on the same 5,881 test images.

<div class="takeaway">
<strong>Cause:</strong> this is a distribution shift failure. The model was trained on clean images and was not exposed to noisy inputs during training. The clean benchmark result therefore does not fully describe performance under degraded real-world conditions.
</div>

<!--
The clean result is the reference point: 99.81 percent accuracy on the original test images. With Gaussian blur, accuracy remains relatively high at 97.01 percent, so moderate blur does not destroy performance. Gaussian noise is very different: accuracy falls to 72.01 percent, which means roughly 1,650 additional errors on the same test set. The most likely explanation is distribution shift, because the model was trained on clean images and never learned to handle this kind of noisy input.
-->

---

## This work is a controlled first step: three directions for improvement

<div class="kicker">Limitations and next steps</div>

<div class="two-col">
  <div>
    <p><strong>Constraints to keep in mind</strong></p>
    <ul>
      <li>Pre-cropped images: the model classifies signs but does not locate them</li>
      <li>We resized all images to 32×32 px; higher resolutions are feasible but require more compute</li>
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
This slide puts the results into perspective. The current model is a strong classifier for cropped GTSRB signs, but it is not yet a full road-scene recognition system. The most direct next step is robustness training, especially adding noise and blur augmentation, because that targets the weakness we just saw. A second step is stronger validation across independent splits, since the model rankings are based on one split and three seeds. Longer term, object detection is necessary, because a real system must first find traffic signs in a live camera frame before it can classify them.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Conclusion

<div class="subtitle">Depth was strongest in the single run. LeakyReLU surprised in multi-seed evaluation. Robustness remains open.</div>

<!--
To conclude, there are three main takeaways from this project. First, compact CNNs already perform extremely well on clean, cropped GTSRB images. Second, architectural choices do matter, but the multi-seed analysis showed that a single run can be misleading when the differences are very small. Third, the most important limitation is robustness: the selected model performs very well under clean benchmark conditions, but noise causes a large drop. That is the key gap between a strong benchmark classifier and a more realistic traffic-sign recognition system.
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
    <p>Depth reduced errors from <strong>30 to 11</strong> in the single run at near-zero cost. Multi-seed analysis then showed LeakyReLU CNN as second-best overall, so the single-run last place was a misleading outlier.</p>
  </div>
  <div class="card">
    <h3>Robustness is the open problem</h3>
    <p>A <strong>27.80 pp</strong> drop under Gaussian noise shows that clean benchmark accuracy does not reflect degraded-input performance. Distribution shift is the primary unresolved challenge.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 20px;">
<strong>The benchmark result is strong on clean images. Multi-seed evaluation is essential to draw reliable conclusions from small accuracy differences. Extending robustness to degraded inputs is the most important next step.</strong>
</div>

<!--
First, we saw that a compact CNN already performs very strongly for this cropped benchmark task. Second, the model comparison illustrated why both architectural choices and repeated evaluation are important for drawing reliable conclusions. Third, the biggest open problem remains robustness to degraded inputs, particularly in the presence of Gaussian noise.
-->

---

<!-- _class: title -->
<!-- _paginate: false -->

# Thank You

## Questions?

<!--
Thank you for your attention. I am happy to answer any questions about the models, the results from different seeds, or the robustness experiments. If you are interested in further details, I also have backup slides covering topics like the autoencoder, hyperparameter search, individual seed runs, and limitations.
-->

---

<!-- _class: divider -->
<!-- _paginate: false -->

# Backup Slides

---

## Backup: Anomaly Detection via Autoencoder

<div class="kicker">Extension · proof of concept</div>

<div class="lead">A standard classifier always assigns one of its 43 known classes, even for inputs it has never seen.</div>

<div class="two-col">
  <div>
    <p><strong>Approach</strong></p>
    <p>A compression autoencoder learns to reconstruct known traffic signs via a 128-dimensional bottleneck. Inputs that do not match the learned reconstruction pattern may produce elevated reconstruction error, which can be used as a candidate anomaly signal.</p>
    <br>
    <p><strong>Threshold</strong></p>
    <p>Set at the 95th percentile of validation reconstruction errors (1.091). By construction, this flags 5% of in-distribution validation images.</p>
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
    <p class="muted" style="font-size: 0.88em;">No true out-of-distribution test set was available. The method is implementable; its real-world detection performance is not yet validated.</p>
  </div>
</div>

<!--
This backup slide shows the autoencoder as a proof of concept rather than a validated anomaly detector. The idea is to use reconstruction error as a candidate anomaly score: images that are harder to reconstruct receive a higher error. The threshold is set at the 95th percentile of validation errors, so 5 percent of normal validation images are flagged by construction. Since no true out-of-distribution test set was used, the actual detection performance remains unvalidated.
-->

---

## Backup: Hyperparameter Sensitivity (Optuna)

<div class="kicker">Extension · 30-trial Bayesian search</div>

**Goal:** verify that manually chosen training settings are not fragile.

| Hyperparameter | Search range | Best value found |
|---|---|:---:|
| Learning rate | 0.0001 – 0.01 | **0.00124** |
| Dropout rate | 0.2 – 0.6 | **0.274** |
| Batch size | 32, 64, 128 | **32** |
| Optimiser | Adam / SGD | **Adam** |
| Weight decay | 1×10⁻⁵ – 1×10⁻² | **0.000698** |

**Best trial (trial 6):** 99.91% validation accuracy

- Adam appeared in all top-5 trials in this search
- Best learning rate (0.00124) is close to our default (0.001)
- Best dropout (0.274) is lower than our default (0.5)

<div class="takeaway">
<strong>Sensitivity check:</strong> the manual defaults appear to be in a reasonable region of the search space, but this was not a full hyperparameter study.
</div>

<!--
This slide is a sensitivity check, not a full hyperparameter study. Optuna searched 30 configurations for the Stride CNN, with each trial limited to 10 epochs. The best learning rate was close to our default, and Adam appeared in all top-5 trials. This supports the idea that our default settings were reasonable, but it does not prove global optimality.
-->

---

## Backup: Multi-Seed Per-Run Results

<div class="kicker">All 15 training runs · seeds 42, 123, 2026</div>

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
This slide shows the raw runs behind the aggregated multi-seed table. It is useful if someone asks why the mean ranking differs from the canonical single-run ranking. For example, Deep CNN drops under seed 2026, which shows why its advantage should be interpreted as an average effect rather than a guaranteed win in every run. LeakyReLU, by contrast, stays very stable across all three seeds, which explains its strong multi-seed ranking.
-->

---

## Backup: Full Limitations

| Limitation | Impact |
|---|---|
| Internal hold-out split only | Not directly comparable to official GTSRB leaderboard |
| Pre-cropped images | Cannot locate signs; classification only, not detection |
| Three-seed validation, no independent splits | Ranking is more reliable than a single run, but still split-dependent |
| Input resized to 32×32 px (preprocessing choice) | Fine details (numerals, symbols) may be partially lost; higher resolution is feasible |
| No noise or blur augmentation during training | Model not robust to degraded inputs |
| GTSRB: German roads, single camera, controlled weather | Limited generalisation to other regions, conditions, or sign standards |
| Rare classes have few test images | A single missed prediction shifts per-class accuracy significantly |

<!--
This slide collects the main methodological limitations. The most important points are the internal hold-out split, the cropped-image scope, the limited number of seeds, and the 32 by 32 input resolution. The robustness gap is also important, because the model was not trained with noise or blur augmentation. These limitations do not invalidate the results, but they define what would need to be tested before moving toward a more realistic system.
-->

---

<!-- _class: image-focus -->

## Backup: t-SNE Feature Space (Deep CNN)

![](results/task05/tsne_feature_space.png)

**2,000 validation samples projected to 2D from the Deep CNN's 512-dimensional internal feature space (perplexity = 30).**

Most classes form distinct clusters. Overlap concentrates among speed limit signs and visually similar warning triangles, consistent with the per-class error pattern.

<!--
This slide visualises the Deep CNN feature space by projecting 2,000 validation samples from 512 dimensions down to two dimensions with t-SNE. Most classes form visually separated clusters, which supports the interpretation that the model learned class-relevant representations. The overlap is mainly visible among speed limit signs and similar warning signs, which is consistent with the error analysis. Since t-SNE is only a projection, this should be treated as supporting visual evidence rather than a formal proof.
-->

---

## Backup: Timing Plan

| Slide | Topic | Time |
|---|---|:---:|
| 1 | Title | 0:20 |
| 2 | Divider: Dataset | 0:10 |
| 3 | Task motivation | 0:40 |
| 4 | Dataset overview | 0:45 |
| 5 | Three dataset properties | 0:45 |
| 6 | Visual ambiguity zoom-in | 0:35 |
| 7 | Class imbalance visual | 0:30 |
| 8 | Divider: Our Approach | 0:10 |
| 9 | Baseline architecture | 0:50 |
| 10 | Baseline result | 0:35 |
| 11 | Architectural questions | 0:45 |
| 12 | Variant mapping | 0:45 |
| 13 | Controlled training setup | 0:40 |
| 14 | Divider: Model Comparison | 0:10 |
| 15 | Single-run results table | 0:45 |
| 16 | Multi-seed validation | 1:00 |
| 17 | Divider: Deep CNN Evaluation | 0:10 |
| 18 | Error profile | 0:40 |
| 19 | Misclassified examples | 0:30 |
| 20 | Confusion matrix | 0:25 |
| 21 | Frequency bias analysis | 0:40 |
| 22 | Grad-CAM | 0:25 |
| 23 | Divider: Robustness | 0:10 |
| 24 | Robustness results | 0:50 |
| 25 | Limitations and next steps | 0:50 |
| 26 | Divider: Conclusion | 0:10 |
| 27 | Summary | 0:50 |
| 28 | Thank You | n/a |
| **Total** | | **~14:45** |

<!--
Use this as a rough guide to stay within 15 minutes. The two highest-risk sections for running over time are Model Comparison and Deep CNN Evaluation, because these are the most technical parts of the presentation. If time is tight, the confusion matrix and Grad-CAM slides can be covered briefly. Backup slides are purely for questions; do not cover them unless someone specifically asks.
-->