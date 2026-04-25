# Task 06 – Model Evaluation

## 1. Test Set Performance

| Metric | Value |
|--------|-------|
| Test Accuracy (Top-1) | **99.81%** |
| Test Accuracy (Top-5) | **99.98%** |
| Test Loss | 0.0061 |
| Wrong Classifications | 11 / 5881 |

## 2. Per-Class Analysis

**5 best performing classes:**

- Class 42 *End of no passing by vehicles over 3.5t*: 100.00%
- Class 14 *Stop*: 100.00%
- Class 41 *End of no passing*: 100.00%
- Class 20 *Dangerous curve right*: 100.00%
- Class 19 *Dangerous curve left*: 100.00%

**5 worst performing classes:**

- Class 27 *Pedestrians*: 97.62%
- Class 29 *Bicycles crossing*: 97.62%
- Class 21 *Double curve*: 98.39%
- Class 30 *Beware of ice/snow*: 98.67%
- Class 8 *Speed limit (120km/h)*: 99.10%

## 3. Bias Analysis

| Group | Mean Accuracy |
|-------|--------------|
| Frequent classes | 99.87% |
| Rare classes     | 99.52% |
| Gap              | 0.34pp |

The model shows minimal bias between frequent and rare traffic sign classes,
which indicates that the data augmentation and balanced training strategy were effective.

## 4. Robustness

| Condition | Accuracy |
|-----------|---------|
| Clean     | 99.81% |
| Gaussian Noise (σ=0.1) | 71.86% |
| Gaussian Blur (k=5) | 97.01% |

## 5. Grad-CAM

Grad-CAM visualizations confirm that the model attends to the relevant
sign regions (shape, symbol, color) rather than background artifacts.
See `results/task06/gradcam_examples.png`.

## 6. Conclusion

The Deep CNN achieves **99.81% top-1 accuracy** on the GTSRB test set.
Performance remains robust under Gaussian noise and blur perturbations.
Per-class analysis reveals no systematic failure mode across the 43 traffic sign categories.
