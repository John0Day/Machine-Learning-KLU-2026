# Machine-Learning-KLU-2026

CNN-based traffic sign classification for the KLU Machine Learning course, using the German Traffic Sign Recognition Benchmark (GTSRB). The project trains and evaluates five CNN architectures, achieving up to **99.81% test accuracy**.

---

## Quick Start

```bash
./scripts/setup_project.sh
```

This single command will:
1. Create a virtual environment (`.venv`) and install all dependencies
2. Download and extract the GTSRB dataset into `data/raw/`
3. Run dataset inspection to generate initial outputs in `results/`

**Options:**

| Flag | Effect |
|------|--------|
| `--skip-data` | Only install dependencies, skip dataset download |
| `--force-data` | Re-download archives from scratch |
| `--keep-zips` | Keep ZIP archives instead of deleting them |
| `--force-install` | Force dependency reinstall |

---

## Project Structure

```
src/
  dataset.py          Dataset inspection and visualisation (Task 02)
  preprocessing.py    DataLoaders, transforms, augmentation (Task 03)
  model.py            Baseline CNN architecture (Task 04)
  model_improved.py   DeepCNN, LeakyReLUCNN, StrideCNN, MobileNetV2 (Task 05)
  train.py            Baseline training script (Task 04)
  train_improved.py   Multi-model training and comparison (Task 05)
  evaluate.py         Full model evaluation with Grad-CAM (Task 06)
  tune.py             Bayesian hyperparameter tuning with Optuna (Task 05)
  autoencoder.py      Convolutional autoencoder for anomaly detection (Task 05)
  visualize.py        t-SNE latent space visualisation (Task 05)
  demo_offline.py     Presentation-ready prediction grids (no webcam)
  demo_ui.py          Interactive Gradio web UI for all models

models/               Trained model weights (.pth)
results/              All generated plots, metrics, and JSON files
tests/                Unit tests
data/                 Raw dataset (excluded from git)
```

---

## Tasks

### Task 02 — Dataset Inspection

```bash
.venv/bin/python src/dataset.py
```

Outputs in `results/task03/`:
- `class_distribution.png` — bar chart of per-class image counts
- `sample_images_by_class.png` — one image per class grid
- `resolution_distribution_top20.png` — top-20 most common resolutions
- `class_mapping.csv` — class ID to name mapping
- `dataset_stats.json` — summary statistics

---

### Task 03 — Preprocessing

Preprocessing is handled automatically by `src/preprocessing.py`. The pipeline applies:
- **Training**: Random rotation (±15°), color jitter, random affine, normalization
- **Validation/Test**: Resize, normalize (deterministic)

Split: **70% train / 15% val / 15% test** with fixed seed 42.

---

### Task 04 — Baseline Model

```bash
.venv/bin/python src/train.py
```

With custom options:
```bash
.venv/bin/python src/train.py --epochs 30 --seed 123 --run-name seed123 \
    --results-dir results/task04 --models-dir models
```

Outputs:
- `models/baseline_seed-42.pth`
- `results/task04/baseline_loss_curve_seed-42.png`
- `results/task04/baseline_history_seed-42.json`

**Baseline CNN**: 3 conv blocks (BatchNorm + ReLU + MaxPool) + FC classifier with Dropout(0.5). 629K parameters. Achieves ~99.29% test accuracy.

---

### Task 05 — Model Improvements

**Train and compare all five model variants:**

```bash
.venv/bin/python src/train_improved.py
```

Outputs in `results/task05/`:
- `model_comparison.json` — accuracy, parameters, training time per model
- `model_comparison_curves.png` — training accuracy curves
- `model_comparison_summary.png` — accuracy vs. parameters vs. training time

| Model | Test Accuracy | Parameters |
|-------|:---:|:---:|
| Baseline CNN | 99.49% | 629K |
| **Deep CNN** ⭐ | **99.81%** | 936K |
| MobileNetV2 | 99.66% | 2.56M |
| LeakyReLU CNN | 99.46% | 629K |
| Stride CNN | 99.52% | 823K |

**Hyperparameter tuning (Optuna):**

```bash
.venv/bin/python src/tune.py --n-trials 30 --epochs 10
```

Outputs in `results/`:
- `best_hyperparams.json`
- `tuning_results.png`

**Latent space visualisation (t-SNE):**

```bash
.venv/bin/python src/visualize.py --model-path models/baseline_seed-42.pth
```

Output: `results/tsne_feature_space.png`

**Autoencoder anomaly detection:**

```bash
# Train and evaluate
.venv/bin/python src/autoencoder.py --mode both --device mps

# Train only
.venv/bin/python src/autoencoder.py --mode train

# Evaluate only (requires trained model)
.venv/bin/python src/autoencoder.py --mode evaluate
```

Outputs in `results/`:
- `autoencoder_loss.png`
- `autoencoder_error_distribution.png`
- `autoencoder_reconstructions.png`

---

### Task 06 — Evaluation

Run full evaluation for a specific model:

```bash
.venv/bin/python src/evaluate.py --model-type deep --device mps
```

Available model types: `baseline`, `deep`, `leaky`, `stride`, `mobilenet`

Outputs in `results/task06/<model>/`:
- `confusion_matrix_normalized.png`
- `per_class_accuracy.png`
- `precision_recall_per_class.png`
- `misclassifications_top_confidence.png`
- `gradcam_examples.png`
- `bias_analysis_mean_accuracy.png`
- `robustness_metrics.json`
- `evaluation_summary.json`
- `classification_report.txt`

---

## Demo

### Offline Demo (no webcam)

```bash
.venv/bin/python src/demo_offline.py --model models/baseline_seed-42.pth
```

Generates prediction grids and confusion matrix in `results/demo_offline/`.

### Interactive Web UI

```bash
# Install Gradio once
pip install gradio

# Launch the UI
python src/demo_ui.py
```

Open `http://localhost:7860` in your browser. Features:
- Upload any traffic sign image
- Select between all 5 trained models
- View top prediction + top-5 confidence bar chart
- Model description and comparison table

For a public shareable link (useful for presentations):
```bash
python src/demo_ui.py --share
```

---

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Report

The full written project report is available at [`report.md`](report.md) (~5000 words).

---

## Data Source

Official GTSRB archives from ERDA:
- `GTSRB_Final_Training_Images.zip`
- `GTSRB_Final_Training_HueHist.zip`
- `GTSRB_Final_Training_HOG.zip`
- `GTSRB_Final_Training_Haar.zip`

Dataset files are excluded from git due to their size.

---

## License

See [LICENSE](LICENSE).
