# Project Tasks - CNN Traffic Sign Classification (GTSRB)

## Branch Strategy

```text
main                          <- stable, completed versions only
`-- dev                       <- integration branch for all features
    |-- task/01-project-setup
    |-- task/02-data-loading
    |-- task/03-data-preprocessing
    |-- task/04-baseline-model
    |-- task/05-model-improvement
    |-- task/06-evaluation
    `-- task/07-report
```

**Workflow per task:**
1. Create a new branch from `dev`: `git checkout -b task/XX-name`
2. Implement and commit changes
3. Open a pull request to `dev`
4. Teammate reviews, then merge

---

## Tasks

---

### Task 01 - Project Setup
**Branch:** `task/01-project-setup`  
**Goal:** Set up the repository foundation

**Work items:**
- Create folder structure:
  ```text
  /data/          <- GTSRB raw data (do not push to repo)
  /src/           <- Python scripts (model.py, dataset.py, train.py, evaluate.py)
  /notebooks/     <- Jupyter notebooks for experiments
  /models/        <- saved model weights (.pth)
  /results/       <- plots, metrics, confusion matrix
  ```
- Add `requirements.txt` with dependencies (torch, torchvision, numpy, matplotlib, seaborn, scikit-learn)
- Add `.gitignore` (data/, models/*.pth, __pycache__, .env)
- Add `README.md` with project description, setup guide, and dataset download link

**Definition of Done:** Repo is cloned, `pip install -r requirements.txt` runs without errors, and structure is in place.

---

### Task 02 - Data Loading
**Branch:** `task/02-data-loading`  
**Goal:** Load and inspect the GTSRB dataset

**Work items:**
- Load GTSRB via `torchvision.datasets.GTSRB`
- Visualize class distribution (bar chart: number of images per class)
- Show sample images from multiple classes
- Build class mapping (index -> traffic-sign name)
- Export dataset statistics (image count, dimensions, resolution distribution)

**Definition of Done:** `src/dataset.py` runs and visualizations are saved to `/results/`.

---

### Task 03 - Data Preprocessing
**Branch:** `task/03-data-preprocessing`  
**Goal:** Prepare images for CNN training

**Work items:**
- Resize all images to a consistent size (32x32 or 64x64 px)
- Normalize pixels: 0-255 -> 0.0-1.0 (with training-set mean/std)
- Data augmentation for training split:
  - Random rotation (+/-15 deg)
  - Random brightness/contrast changes
  - Random horizontal flip (only where semantically valid)
- Train/Validation/Test split (for example 70% / 15% / 15%)
- PyTorch `DataLoader` with batches (for example batch size 64)

**Definition of Done:** Train/Val/Test `DataLoader`s run and batch shape is verified.

---

### Task 04 - Baseline Model
**Branch:** `task/04-baseline-model`  
**Goal:** Implement and train the first working CNN baseline

**Architecture (baseline):**
```text
Input (3 x 32 x 32)
-> Conv(32 filters, 3x3) + ReLU + MaxPool(2x2)
-> Conv(64 filters, 3x3) + ReLU + MaxPool(2x2)
-> Flatten
-> Linear(256) + ReLU + Dropout(0.5)
-> Linear(43)  <- 43 classes
-> Softmax
```

**Work items:**
- Define model in `src/model.py` (PyTorch `nn.Module`)
- Implement training loop in `src/train.py` (Adam optimizer, cross-entropy loss)
- Log validation loss and accuracy after each epoch
- Plot loss curves (train vs validation)
- Save model weights (`models/baseline.pth`)

**Definition of Done:** Model trains without errors and can reach validation accuracy > 80%.

---

### Task 05 - Model Improvement
**Branch:** `task/05-model-improvement`  
**Goal:** Improve the baseline and compare variants

**Work items:**
- Variant A: deeper network (4-5 conv layers)
- Variant B: batch normalization after every conv layer
- Variant C: transfer learning with pretrained MobileNet (`torchvision.models`)
- Learning-rate scheduling (for example `StepLR` or `ReduceLROnPlateau`)
- Comparison table: accuracy / training time / model size
- Save best model

**Definition of Done:** At least 3 variants are compared and results are stored in `results/model_comparison.csv`.

---

### Task 06 - Evaluation
**Branch:** `task/06-evaluation`  
**Goal:** Evaluate the model thoroughly and analyze weak points

**Work items:**
- Test-set accuracy of the best model
- Confusion matrix (heatmap): which classes are confused?
- Precision, recall, F1-score per class (`sklearn.metrics.classification_report`)
- **Bias analysis:** compare accuracy for frequent vs rare classes
- Visualize examples of misclassified images
- Grad-CAM visualization: which image regions activate the model?
- Robustness test: evaluate on noisy/blurred images

**Definition of Done:** All plots and metrics are generated in `/results/` and are reproducible.

---

### Task 07 - Report
**Branch:** `task/07-report`  
**Goal:** Write the final report (3000-5000 words)

**Work items:**
- Create report as `report.md` or `report.pdf` at repo root
