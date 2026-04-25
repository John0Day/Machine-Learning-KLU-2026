# Task 06 – Model Comparison

| Model | Top-1 Acc | Top-5 Acc | Loss | Noisy Acc | Blurred Acc | Wrong |
|-------|----------|----------|------|-----------|------------|-------|
| Baseline CNN | 99.49% | 100.00% | 0.0203 | 61.62% | 95.56% | 30 |
| Deep CNN | 99.81% | 99.98% | 0.0061 | 71.86% | 97.01% | 11 |
| MobileNetV2 | 99.66% | 100.00% | 0.0140 | 70.84% | 98.33% | 20 |
| LeakyReLU CNN | 99.46% | 99.98% | 0.0160 | 60.98% | 96.72% | 32 |
| Stride CNN | 99.52% | 99.98% | 0.0173 | 81.14% | 97.36% | 28 |

> Best model: **Deep CNN** — highest Top-1 accuracy (99.81%).

*Noise std = 0.1, Blur kernel = 5*
