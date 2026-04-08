# Machine-Learning-KLU-2026

Traffic sign classification project for the KLU Machine Learning course, based on the German Traffic Sign Recognition Benchmark (GTSRB).

## Quick Start (One Command)

```bash
./scripts/setup_project.sh
```

This single command will:

1. Create a local virtual environment (`.venv`)
2. Install Python dependencies from `requirements.txt`
3. Download and extract the official GTSRB training archives into `data/raw/`
4. Run dataset inspection (`src/dataset.py`) to generate initial outputs in `results/`

By default it prefers `python3.12`, then `python3.11`, then `python3.10`, then `python3`.
Running it again is safe: unchanged dependencies are skipped, verified archives are reused, and extraction uses a cache marker.

## Project Structure

```text
/data/          dataset files (ignored by git)
/src/           Python source files (dataset, model, train, evaluate)
/notebooks/     Jupyter notebooks for experiments
/models/        trained model weights (.pth/.pt) (ignored by git)
/results/       plots, metrics, confusion matrices (ignored by git)
```

## Data Source

The dataset pull script uses these official ERDA URLs by default:

- `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip`
- `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HueHist.zip`
- `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HOG.zip`
- `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Haar.zip`

These files are intentionally excluded from git because of their size.

## Maintainer Checksum Step (One-Time)

Before sharing the repo with course staff, generate and commit checksums once:

```bash
./scripts/fetch_gtsrb.sh --skip-verify --write-checksums
```

Then commit `checksums/gtsrb.sha256` so downloads can be verified automatically.

## Dataset Inspection (Task 02)

Run:

```bash
.venv/bin/python src/dataset.py
```

This generates:

- `results/class_distribution.png`
- `results/sample_images_by_class.png`
- `results/resolution_distribution_top20.png`
- `results/class_mapping.csv`
- `results/dataset_stats.json`

## Baseline Training (Task 04)

Run baseline model training:

```bash
.venv/bin/python src/train.py --epochs 10 --batch-size 64
```

Run with an explicit run label (prevents output overwrite):

```bash
.venv/bin/python src/train.py --epochs 10 --batch-size 64 --seed 123 --run-name seed123
```

Outputs:

- `models/baseline_<run_name>.pth` (default run name: `seed-<seed>`)
- `results/baseline_loss_curve_<run_name>.png`
- `results/baseline_history_<run_name>.json`

## Offline Demo (No Camera)

Generate presentation-ready prediction visuals from the test split:

```bash
.venv/bin/python src/demo_offline.py --model models/baseline_seed-123.pth --device cpu
```

Demo outputs are written to `results/demo_offline/`:

- `predictions_grid.png`
- `misclassifications_top_confidence.png`
- `confusion_matrix.png`
- `summary.json`

## Optional Commands

- Override dataset base URL:

```bash
./scripts/setup_project.sh --base-url "https://your-host.example.com/gtsrb"
```

- Skip dataset fetch (only install dependencies):

```bash
./scripts/setup_project.sh --skip-data
```

- Re-download archives from scratch:

```bash
./scripts/setup_project.sh --force-data
```

- Force dependency reinstall:

```bash
./scripts/setup_project.sh --force-install
```

- Verify already downloaded archives only:

```bash
./scripts/fetch_gtsrb.sh --verify-only
```

- Run local downloader safety test:

```bash
./scripts/test_fetch_gtsrb.sh
```

## License

See [LICENSE](LICENSE).
