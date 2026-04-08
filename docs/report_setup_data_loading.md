# Project Setup and Data Loading

## Overview

This report documents the implementation of **Task 01 (Project Setup)** and **Task 02 (Data Loading)** for the GTSRB traffic sign classification project. The objective of these two tasks is to create a reproducible project foundation and to establish a transparent, inspectable data-loading workflow before model training starts.

---

## Task 01 – Project Setup

### Repository Structure

A clear project structure was created to separate raw data, source code, experiments, model artifacts, and generated outputs:

```text
/data/
/src/
/notebooks/
/models/
/results/
/docs/
/scripts/
/checksums/
```

This structure makes responsibilities explicit and prevents coupling between data handling, experimentation, and model lifecycle files.

### Dependency Management

A central `requirements.txt` was introduced to standardize the Python environment across contributors and machines. This ensures that all team members and course staff can reproduce the same runtime context.

### Git Hygiene and Large File Handling

A strict `.gitignore` was configured so that large datasets and generated artifacts are not committed accidentally. In particular:

- raw dataset storage under `data/raw/` is excluded
- model weights and generated result files are excluded
- cache and temporary build artifacts are excluded

This keeps the repository lightweight and avoids unstable history caused by binary artifacts.

### Reproducible Startup Workflow

A one-command bootstrap script was introduced:

```bash
./scripts/setup_project.sh
```

It automates environment setup, dependency installation, dataset retrieval/extraction, and dataset inspection. The script is designed to be safe on reruns (idempotent behavior): unchanged dependencies are skipped, verified files are reused, and extraction is cached.

### Dataset Retrieval Pipeline

A dedicated fetch script (`scripts/fetch_gtsrb.sh`) was added with the official ERDA archive URLs as defaults. It supports:

- checksum verification (`SHA-256`)
- extraction to `data/raw/`
- safe reruns with lock/caching logic
- optional ZIP cleanup to trash after extraction

This removes manual setup steps and reduces onboarding errors.

---

## Task 02 – Data Loading

### Dataset Input Format

The data loader targets the official GTSRB training image archive layout:

- class folders `00000` to `00042`
- class-level annotation files `GT-xxxxx.csv`
- PPM image files per class

The loader parses all class CSV files and builds a unified in-memory record list containing class id, class name, image dimensions, and image path.

### Class Mapping

A complete class-id to sign-name mapping for all 43 classes was implemented and exported to:

- `results/class_mapping.csv`

This mapping is used both for interpretability in plots and for downstream reporting.

### Dataset Statistics and Inspection Artifacts

The script `src/dataset.py` generates reproducible inspection outputs in `results/`:

- `class_distribution.png`
- `sample_images_by_class.png`
- `resolution_distribution_top20.png`
- `dataset_stats.json`

These outputs provide immediate visibility into class imbalance, visual class examples, and image-size heterogeneity.

### Key Observations from Current Run

From the executed pipeline:

- total training images: **39,209**
- number of classes: **43**
- image dimensions are highly variable, which justifies explicit preprocessing before model training

This confirms that Task 02 goals (loading, exploration, and export of diagnostics) are fulfilled.

---

## Reproducibility and Integrity

To guarantee that collaborators use identical dataset files, SHA-256 hashes were added in:

- `checksums/gtsrb.sha256`

During normal verified fetch mode, each archive is hashed and validated before extraction. This prevents silent corruption and strengthens reproducibility for grading and collaboration.

---

## Summary

Task 01 established a robust engineering baseline for the project: structure, dependency control, safe data handling, and one-command setup. Task 02 implemented a complete data-loading and inspection pipeline for GTSRB, including class mapping, dataset statistics, and visual diagnostics. Together, these tasks provide a reproducible and auditable foundation for the next modeling stages.
