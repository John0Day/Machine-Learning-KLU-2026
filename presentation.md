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

Professor question catalogue:
Q: Why start the approach section with preprocessing?
A: Because preprocessing defines the exact input distribution for every model. It determines image size, augmentation, normalisation, and the deterministic validation and test setup.
Q: Why not start directly with the model architecture?
A: The architecture can only be interpreted properly once it is clear what the model receives as input. The 32 by 32 input size also affects the later discussion of fine visual details.
Q: What is the transition from this section?
A: We move from dataset challenges to controlled model inputs, then to the baseline model that serves as the reference point.
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
Before any model sees the data, every image goes through the same preprocessing pipeline. First, we resize the image to 32 by 32 pixels. This resizing step is our preprocessing choice, because the original GTSRB images have different sizes and need to be converted into one fixed input format for the CNN. During training, we then apply augmentation to simulate realistic variation, such as tilted camera angles, lighting changes, and slightly off-centre signs. For validation and testing, we only resize and normalise the images. There are no random transforms, so every model is evaluated on the same inputs.

Professor question catalogue:
Q: Why did you resize to 32 by 32 pixels?
A: We needed a fixed input size for all CNN models. 32 by 32 keeps training efficient and makes the model comparison easier. The trade-off is that small numerals and symbols can become harder to distinguish.
Q: Is 32 by 32 a dataset constraint?
A: No. It is our preprocessing choice. The original GTSRB images have different sizes, so they must be resized before they can be passed into a fixed CNN architecture.
Q: Why use augmentation only for training?
A: Augmentation helps the model learn from slightly varied examples. Validation and test images stay deterministic, so the evaluation remains reproducible and fair across models.
Q: Could 64 by 64 improve results?
A: Possibly. A higher resolution could preserve fine details better, especially speed-limit numerals. But it would also increase compute, so it would need to be tested empirically.
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

Professor question catalogue:
Q: Why include this visual example?
A: It makes the preprocessing concrete. Instead of only listing augmentation operations, the slide shows what the model actually sees during training.
Q: Are these images used for validation or testing?
A: No. These are augmented training examples. Validation and test images are not randomly augmented.
Q: Does augmentation make the task artificially harder?
A: It makes training more varied, but the goal is to improve generalisation. The evaluation remains on deterministic validation and test images.
Q: Why not use stronger augmentation?
A: Stronger augmentation could help robustness, especially for noise and blur, but overly strong transformations may distort the sign identity. This would need tuning.
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
The baseline model processes images in three successive stages. The first stage learns basic visual patterns such as edges and colour boundaries, the second combines these into shapes and contours, and the third builds more sign-specific representations such as numerals inside circles. After these stages, the model compresses the image into a 2,048-value feature representation and uses it to predict one of the 43 classes. With 629,000 parameters and training from scratch on GTSRB, this is our reference point for everything that follows.

Professor question catalogue:
Q: Why is a compact CNN a reasonable baseline?
A: The task uses small, cropped traffic sign images, so a compact CNN can learn useful hierarchical visual features without requiring a very large model.
Q: Why train from scratch instead of using a pretrained model immediately?
A: GTSRB is narrow and structured, and there are enough labelled images for a compact CNN to learn task-specific features. Pretraining is tested later as a separate variant.
Q: What does the 2,048-value representation mean?
A: It is the flattened output after the three convolutional blocks: 128 feature maps at 4×4 spatial resolution, giving 128×4×4 = 2,048 values. This vector is then compressed to 256 units by the first fully connected layer before the final 43-class output layer. So the full classifier path is 2,048 → 256 → 43.
Q: Why explain the baseline before showing results?
A: The baseline is the reference point for all later model changes. The audience needs to understand what is being improved or compared.
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

Professor question catalogue:
Q: Why continue if the baseline already reaches 99.49 percent?
A: The remaining 30 errors are still informative. They show where the model struggles and motivate testing whether architecture choices can reduce those mistakes.
Q: Is 99.49 percent directly comparable to the official GTSRB benchmark?
A: No. This is on our internal split, so it is internally comparable across our models but not directly comparable to the official leaderboard.
Q: Why mention parameters here?
A: Parameter count shows that the baseline is not only accurate but also compact. This matters when comparing later variants that may be larger or slower.
Q: What should the audience take from this slide?
A: The baseline is already strong, so later improvements must be interpreted carefully rather than assumed to be large or obvious.
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

Professor question catalogue:
Q: Were the four variants chosen randomly?
A: No. Each variant corresponds to a specific architectural question raised by the baseline result and dataset properties.
Q: Why test more depth?
A: Additional depth may help learn more abstract sign-level features and better separate fine details such as numerals.
Q: Why test transfer learning?
A: It tests whether general visual features learned from ImageNet transfer to a narrow traffic sign domain.
Q: Why test activation and downsampling?
A: Activation choice may affect training stability, and learned downsampling tests whether fixed pooling discards useful spatial detail.
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

Professor question catalogue:
Q: Does each variant change exactly one thing?
A: Each variant targets one main design aspect, but other properties such as parameter count, training time, and optimisation dynamics can also change.
Q: Why is MobileNetV2 included if the project focuses on compact CNNs?
A: It was included deliberately as an upper-bound candidate for transfer learning. MobileNetV2 brings ImageNet-pretrained features and 2.56M parameters — roughly four times the baseline. The result that it shows no stable accuracy advantage despite that cost is itself a finding: it tells us that ImageNet pretraining does not transfer meaningfully to this narrow, structured domain, and that a purpose-built compact CNN is more efficient here.
Q: Why is LeakyReLU included?
A: It tests whether keeping a small gradient for negative activations improves training stability across seeds.
Q: How should this table be interpreted?
A: It is a design map. The performance interpretation comes later and must include accuracy, stability, parameter count, and training time.

Additional detail per variant (if the professor asks for specifics):
- Deep CNN: this is the baseline made slightly deeper. We add one extra convolutional block with 256 filters and increase the hidden fully connected layer from 256 to 512 units. Because there is one more pooling step, the feature map becomes smaller before the classifier. The parameter count increases from 629K to 936K.
- MobileNetV2: this model uses a MobileNetV2 network that was already pretrained on ImageNet. We replace the original ImageNet output layer with a new classifier for the 43 GTSRB traffic sign classes and then fine-tune the model on GTSRB. It is much larger than the baseline, with 2.56M parameters.
- LeakyReLU CNN: this keeps the same architecture as the baseline and has the same parameter count of 629K. The only main change is the activation function: ReLU is replaced by LeakyReLU with a small negative slope of 0.01. This means negative activations are not completely set to zero, which can help gradients keep flowing.
- Stride CNN: this keeps the same basic structure as the baseline, but replaces fixed MaxPool downsampling with strided convolution. In simple terms, the model learns how to reduce the image size instead of using a fixed pooling rule. Because these strided convolutions have learnable weights, the parameter count increases to 823K.
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

Professor question catalogue:
Q: Can all performance differences be attributed only to architecture?
A: Not exclusively. The fixed setup makes model design the main factor, but parameter count, random seed, training dynamics, and early stopping still matter.
Q: Why use early stopping?
A: It stops training when validation loss stops improving, which helps reduce overfitting and avoids training longer just to fit noise.
Q: Why is validation/test preprocessing deterministic?
A: It ensures that each model is evaluated on the same inputs every time, so differences are not caused by random transformations.
Q: Why call this a controlled comparison rather than a tournament?
A: The goal is to test architectural hypotheses under fixed conditions, not simply to find a leaderboard winner.
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
After the model comparison, we now focus on the selected Deep CNN. It has the strongest average accuracy-cost tradeoff, so it is the most relevant model to analyse in detail. The next slides move from model selection to model behaviour.

Professor question catalogue:
Q: Why focus the detailed evaluation only on Deep CNN?
A: Deep CNN was selected after the model comparison because it had the strongest average accuracy-cost tradeoff. This section is not meant to repeat the comparison, but to understand the selected model's behaviour.
Q: Why not analyse LeakyReLU in the same detail?
A: LeakyReLU is a strong alternative, especially because of its stability, but Deep CNN had the highest mean accuracy and strongest single-run improvement. A full follow-up could analyse both models side by side.
Q: What changes in this section compared with model comparison?
A: The focus shifts from which model performs best to where the selected model still makes mistakes, whether the errors have a pattern, and how interpretable the predictions look.
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
The Deep CNN makes only 11 errors on the test set, so we can inspect the remaining mistakes quite concretely. The weaker classes shown here are visually difficult cases, especially where small numerals or symbols matter at 32 by 32 resolution. The pattern suggests that the model is not failing broadly, but mainly struggles with visually similar class boundaries.

Professor question catalogue:
Q: Does this mean the model is unreliable?
A: No. The error count is very small: 11 wrong predictions out of 5,881 test images. The important point is that the remaining errors are not random, but concentrated around visually similar classes.
Q: Why show only these classes?
A: These classes illustrate the weakest or most interpretable error cases. They show where the remaining mistakes occur and why those mistakes are visually plausible.
Q: What does this say about the 32 by 32 input size?
A: It suggests that small internal details can become difficult to resolve at this resolution. This is especially relevant for speed-limit numerals and similar warning signs.
Q: Is this evidence of a general model failure?
A: No. It points to a specific limitation near visually similar class boundaries, not broad failure across the task.
-->

---

<!-- _class: image-focus -->

## Misclassified examples make the error pattern visible

![](results/task06/deep/misclassifications_top_confidence.png)

**Representative examples of the 11 errors: mostly speed limits differing by one digit and warning triangles with very similar silhouettes.**

<!--
The previous slide described the error pattern numerically. Here, the same pattern becomes visible in the actual images. The relevant differences are often small and hard to separate after resizing, which supports the interpretation from the error table.

Professor question catalogue:
Q: Why include actual misclassified examples?
A: They make the error pattern tangible. Instead of only reporting class-level metrics, we can see what the model confused and why the confusion is visually understandable.
Q: What do these examples show?
A: Most examples involve signs that differ by small numerals or have very similar silhouettes. This supports the idea that the model struggles mainly with fine visual distinctions.
Q: What would likely reduce these mistakes?
A: A higher input resolution, such as 64 by 64, could give the model more pixels for the decisive numerals and symbols. This would need to be tested empirically.
Q: Does this prove that 32 by 32 is the only problem?
A: No. It is a plausible contributing factor, but other factors such as class similarity, lighting, blur, and data imbalance can also play a role.
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
After looking at individual examples, the confusion matrix gives the class-level view. The strong diagonal shows that errors are rare overall, and the few off-diagonal entries are not randomly spread across classes. They mostly align with the visually similar sign groups discussed before.

Professor question catalogue:
Q: Why include a confusion matrix if there are only 11 errors?
A: Because it shows whether the errors are scattered randomly or concentrated in specific class pairs. Here, the pattern supports the earlier error analysis.
Q: What does the strong diagonal mean?
A: It means most classes are predicted correctly most of the time. The model is not broadly confused across the 43 classes.
Q: What does Top-5 accuracy add here?
A: It shows that even when the top prediction is wrong, the correct class is almost always still among the model's most likely alternatives.
Q: Should we overinterpret the confusion matrix?
A: No. With only 11 errors, the matrix is mainly supporting evidence, not a full statistical error study.
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
After looking at where the errors occur, we now ask whether rare classes are systematically worse. The gap between the 10 most frequent and 10 rarest classes is only 0.34 percentage points in this run. That suggests no strong class-frequency bias under this setup, but rare classes have few test images, so the result should be interpreted cautiously.

Professor question catalogue:
Q: Does this prove there is no class-frequency bias?
A: No. It suggests no strong frequency bias under this setup, but the rare classes have few test images, so the estimate is uncertain.
Q: Why compare the 10 most frequent and 10 rarest classes?
A: It gives a simple diagnostic for whether high overall accuracy hides weaker performance on underrepresented classes.
Q: Could augmentation explain the small gap?
A: It may help, but we did not isolate its effect with an ablation study. Therefore, we should not claim that augmentation caused the small gap.
Q: What would make this analysis stronger?
A: More seeds, independent splits, confidence intervals, or a stratified evaluation across multiple test samples would make the conclusion more robust.
-->

---

<!-- _class: image-focus -->

## Grad-CAM suggests that predictions rely on sign-relevant regions

![](results/task06/deep/gradcam_examples.png)

**Activation maps concentrate on the sign shape and internal symbol, consistent with task-relevant feature learning.**

<!--
After error and frequency-bias analysis, Grad-CAM gives a qualitative look at which image regions influence the model. The activation maps mostly focus on the sign shape and internal symbol, rather than obvious background areas. This supports the interpretation that the model uses task-relevant visual information, but it is not a formal proof of the decision process.

Professor question catalogue:
Q: Does Grad-CAM prove how the model makes decisions?
A: No. Grad-CAM is qualitative supporting evidence, not a formal causal proof of the decision process.
Q: Why include Grad-CAM anyway?
A: It helps check whether the model appears to focus on sign-relevant regions rather than obvious background shortcuts.
Q: What should we conclude from these maps?
A: The maps support the interpretation that predictions are linked to the sign shape and internal symbols, which is consistent with task-relevant feature learning.
Q: What are the limitations of Grad-CAM?
A: It is coarse, depends on the selected layer, and can highlight regions correlated with the prediction without proving causality.
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

<div class="subtitle">Strong clean-image performance is only the starting point. Stability and robustness decide how reliable the model is.</div>

<!--
To conclude, the main message is not simply that the model reaches high accuracy. The stronger point is that clean benchmark performance has to be interpreted together with evaluation stability and robustness. In our results, compact CNNs perform very well on cropped GTSRB images, but small accuracy differences require multi-seed validation, and degraded inputs expose the main limitation.
-->

---

## Three findings from this experiment

<div class="kicker">Summary</div>

<div class="cards">
  <div class="card">
    <h3>Clean classification works very well</h3>
    <p>A compact 629K-parameter baseline already reaches <strong>99.49%</strong> on the internal test split. This shows that cropped GTSRB classification is highly learnable with a relatively small CNN.</p>
  </div>
  <div class="card">
    <h3>Single runs can mislead</h3>
    <p>Deep CNN reduced errors from <strong>30 to 11</strong> in the single run and had the best mean accuracy. But multi-seed validation showed that LeakyReLU was almost as accurate and more stable.</p>
  </div>
  <div class="card">
    <h3>Robustness is the key limitation</h3>
    <p>Gaussian noise caused a <strong>27.80 pp</strong> accuracy drop. The next step is therefore not only higher clean accuracy, but training and evaluating under degraded input conditions.</p>
  </div>
</div>

<div class="takeaway" style="margin-top: 20px;">
<strong>Final takeaway:</strong> the selected model is a strong benchmark classifier, but a realistic traffic-sign recognition system needs stable validation, robustness training, and eventually object detection for full road scenes.
</div>

<!--
The first finding is that the classification task is highly learnable under clean, cropped benchmark conditions. The second finding is methodological: when all models are above 99 percent, a single run is not enough to draw reliable conclusions. The third finding is practical: robustness is the main gap between a strong benchmark classifier and a more realistic traffic-sign recognition system. That is why the natural next steps are robustness training, stronger validation, and eventually object detection for full road scenes.
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

<div class="lead" style="font-size: 1.12em; margin-bottom: 12px;">
A standard classifier always predicts one of the 43 known classes, even for unfamiliar inputs.
</div>

<div class="two-col" style="grid-template-columns: 54% 46%; gap: 22px; align-items: start; margin-top: 10px;">
  <div>
    <div class="card" style="margin-bottom: 12px;">
      <h3>Idea</h3>
      <p style="font-size: 0.9em;">A compression autoencoder learns to reconstruct known traffic signs through a <strong>128-dimensional bottleneck</strong>. Inputs that do not match the learned reconstruction pattern may produce higher reconstruction error.</p>
    </div>
    <div class="takeaway" style="margin-top: 0; font-size: 0.88em;">
      <strong>Important limitation:</strong> no true out-of-distribution test set was available. This is a candidate anomaly signal, not a validated anomaly detector.
    </div>
  </div>
  <div>
    <table style="font-size: 0.76em;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Latent size</td><td>128 dimensions</td></tr>
      <tr><td>Final reconstruction loss</td><td>0.5118</td></tr>
      <tr><td>Threshold rule</td><td>95th percentile</td></tr>
      <tr><td>Anomaly threshold</td><td>1.091</td></tr>
      <tr><td>Images flagged</td><td>294 / 5,881 (5%)</td></tr>
    </table>
    <div class="takeaway" style="margin-top: 12px; font-size: 0.86em;">
      <strong>Threshold:</strong> by construction, the 95th percentile threshold flags 5% of in-distribution validation images.
    </div>
  </div>
</div>

<!--
This backup slide shows the autoencoder as a proof of concept rather than a validated anomaly detector. The idea is to use reconstruction error as a candidate anomaly score: images that are harder to reconstruct receive a higher error. The threshold is set at the 95th percentile of validation errors, so 5 percent of normal validation images are flagged by construction. Since no true out-of-distribution test set was used, the actual detection performance remains unvalidated.
-->

---

## Backup: Hyperparameter Sensitivity (Optuna)

<div class="kicker">Extension · 30-trial Bayesian search</div>

<div class="lead" style="font-size: 1.08em; margin-bottom: 12px;">
Goal: check whether the manual training settings are in a reasonable range.
</div>

<div class="two-col" style="grid-template-columns: 58% 42%; gap: 22px; align-items: start; margin-top: 8px;">
  <div>
    <table style="font-size: 0.74em;">
      <tr><th>Hyperparameter</th><th>Search range</th><th>Best value</th></tr>
      <tr><td>Learning rate</td><td>0.0001 to 0.01</td><td><strong>0.00124</strong></td></tr>
      <tr><td>Dropout rate</td><td>0.2 to 0.6</td><td><strong>0.274</strong></td></tr>
      <tr><td>Batch size</td><td>32, 64, 128</td><td><strong>32</strong></td></tr>
      <tr><td>Optimiser</td><td>Adam / SGD</td><td><strong>Adam</strong></td></tr>
      <tr><td>Weight decay</td><td>1×10⁻⁵ to 1×10⁻²</td><td><strong>0.000698</strong></td></tr>
    </table>
  </div>
  <div>
    <div class="kpi" style="padding: 14px 12px; margin-bottom: 12px;">
      <div class="number" style="font-size: 1.65em;">99.91%</div>
      <div class="label">best validation accuracy<br>trial 6</div>
    </div>
    <div class="card" style="font-size: 0.86em; padding: 12px 14px;">
      <h3>What it suggests</h3>
      <p>Adam appeared in all top-5 trials. The best learning rate is close to our default of 0.001, while the best dropout is lower than our default of 0.5.</p>
    </div>
  </div>
</div>

<div class="takeaway" style="font-size: 0.9em; margin-top: 14px;">
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

## Backup: t-SNE Feature Space (Deep CNN)

<div class="two-col" style="grid-template-columns: 64% 36%; gap: 24px; align-items: center; margin-top: 8px;">
  <div style="display: flex; justify-content: center; align-items: center;">
    <img src="results/task05/tsne_feature_space.png" style="max-width: 100%; max-height: 56vh; object-fit: contain;" />
  </div>
  <div>
    <div class="lead" style="font-size: 1.02em; line-height: 1.28; margin-bottom: 14px;">
      2,000 validation samples projected from the Deep CNN's 512-dimensional feature representation into 2D.
    </div>
    <div class="takeaway" style="font-size: 0.9em; margin-top: 0;">
      <strong>Interpretation:</strong> most classes appear as separate groups. The visible overlap is mainly among speed limit signs and similar warning signs, which matches the error patterns seen earlier.
    </div>
  </div>
</div>

<!--
This backup slide gives a visual check of the feature space learned by the Deep CNN. Each point is one validation image, and t-SNE projects the 512-dimensional internal representation into two dimensions so we can inspect it visually. The main takeaway is simple: many classes appear separated, while the overlap is mostly where we would expect it, especially among speed limit signs and similar warning signs. This supports the error analysis, but it should not be treated as proof, because t-SNE is only a two-dimensional projection.
-->

