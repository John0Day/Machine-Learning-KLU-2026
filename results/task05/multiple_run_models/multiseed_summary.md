# Task 05 Multi-Seed Comparison

## 1) Per-run results (all models, all seeds)

| Seed | Model | Test Acc | Test Loss | Best Val Acc | Params | Train Time | Epochs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Baseline CNN | 99.20% | 0.0323 | 99.46% | 629,291 | 225.0s | 16 |
| 42 | Deep CNN | 99.78% | 0.0077 | 99.74% | 936,235 | 292.7s | 20 |
| 42 | MobileNetV2 | 99.44% | 0.0197 | 99.54% | 2,562,859 | 529.9s | 20 |
| 42 | Stride CNN | 99.46% | 0.0178 | 99.56% | 823,051 | 295.1s | 20 |
| 123 | Baseline CNN | 99.64% | 0.0104 | 99.81% | 629,291 | 276.0s | 20 |
| 123 | Deep CNN | 99.83% | 0.0075 | 99.74% | 936,235 | 288.9s | 20 |
| 123 | MobileNetV2 | 99.66% | 0.0105 | 99.73% | 2,562,859 | 528.6s | 20 |
| 123 | Stride CNN | 99.30% | 0.0253 | 99.35% | 823,051 | 207.5s | 14 |
| 2026 | Baseline CNN | 99.69% | 0.0118 | 99.73% | 629,291 | 280.6s | 20 |
| 2026 | Deep CNN | 99.46% | 0.0190 | 99.56% | 936,235 | 216.4s | 15 |
| 2026 | MobileNetV2 | 99.20% | 0.0299 | 99.37% | 2,562,859 | 528.8s | 20 |
| 2026 | Stride CNN | 99.59% | 0.0120 | 99.63% | 823,051 | 300.2s | 20 |

## 2) Aggregated comparison (mean ± std over 3 seeds)

| Rank | Model | Test Acc (mean ± std) | Test Loss (mean ± std) | Best Val Acc (mean ± std) | Params | Train Time (mean ± std) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Deep CNN | 99.69% ± 0.17% | 0.0114 ± 0.0054 | 99.68% ± 0.09% | 936,235 | 266.0s ± 35.1s |
| 2 | Baseline CNN | 99.51% ± 0.22% | 0.0182 ± 0.0100 | 99.67% ± 0.15% | 629,291 | 260.5s ± 25.2s |
| 3 | Stride CNN | 99.45% ± 0.12% | 0.0184 ± 0.0054 | 99.51% ± 0.12% | 823,051 | 267.6s ± 42.5s |
| 4 | MobileNetV2 | 99.43% ± 0.19% | 0.0200 ± 0.0079 | 99.55% ± 0.15% | 2,562,859 | 529.1s ± 0.6s |

## 3) Interpretation

- **Best average accuracy:** Deep CNN (99.69%)
- **Best compute efficiency:** Baseline CNN (strong accuracy at lowest parameter count and near-lowest runtime)
- **Most expensive model:** MobileNetV2 (~2.56M params, ~529s/run) without an accuracy advantage over Deep CNN
- **Stability note:** all models are high-performing; differences are small, but Deep CNN keeps the highest average across seeds
