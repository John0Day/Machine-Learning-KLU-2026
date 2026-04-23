# Data Preprocessing

## Overview

Before any image can be fed into a neural network, it must be transformed into a standardized numerical format that the model can process consistently. This section describes the preprocessing pipeline applied to the GTSRB dataset, including the rationale behind each design decision.

---

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) consists of 39,209 training images across 43 classes of traffic signs. The images are provided in PPM format and vary significantly in size, ranging from as small as 15×15 pixels to over 250×250 pixels, and were captured under diverse real-world conditions including varying lighting, occlusion, and motion blur.

---

## Data Splitting

The full training set is divided into three non-overlapping subsets:

- **Training set (70%)** – used to optimize the model weights
- **Validation set (15%)** – used to monitor generalization during training and detect overfitting
- **Test set (15%)** – held out entirely and only used for the final evaluation

The split is performed with a fixed random seed (42) to ensure reproducibility. Separating a dedicated test set is critical for obtaining an unbiased estimate of model performance: evaluating on data that was seen during training or hyperparameter tuning would lead to overly optimistic results. This three-way split is a standard practice in supervised learning to ensure trustworthy results.

---

## Image Transforms

All images pass through a sequence of transformations before being fed to the model. Different transformations are applied depending on whether the image belongs to the training or the validation/test split.

### Resize

Every image is resized to a fixed resolution of 32×32 pixels. This is necessary because convolutional neural networks require all inputs to have the same spatial dimensions, as the weight matrices of fully connected layers have a fixed size. The value of 32×32 is a common choice for GTSRB and offers a good trade-off between computational cost and retaining sufficient visual detail for classification.

### Normalization

Pixel values are first converted from integers in the range [0, 255] to floating-point values in [0, 1] via `ToTensor()`. They are then normalized using the mean and standard deviation computed from the GTSRB training set:

```
mean = (0.3337, 0.3064, 0.3171)
std  = (0.2672, 0.2564, 0.2629)
```

Normalization centers the input distribution around zero and scales it to unit variance. This is important for gradient-based optimization: without normalization, large differences in pixel value scales across channels can cause the loss surface to be poorly conditioned, leading to slow or unstable convergence during gradient descent.

### Data Augmentation (Training Only)

Data augmentation artificially increases the effective size and diversity of the training set by applying random transformations to each image at every epoch. This acts as a regularization technique, reducing overfitting by preventing the model from memorizing specific pixel patterns. The following augmentations are applied only to training images:

| Augmentation | Parameters | Rationale |
|---|---|---|
| Random Rotation | ±15° | Traffic signs may appear slightly tilted due to camera angle or mounting |
| Color Jitter | brightness ±0.4, contrast ±0.4, saturation ±0.3 | Simulates varying lighting and weather conditions |
| Random Affine | translate ±10% | Accounts for signs not being perfectly centered in the image |

Augmentation is deliberately **not** applied to the validation and test sets. These sets must reflect real-world inputs without artificial modifications so that the measured accuracy is an honest estimate of how the model would perform in deployment.

---

## Mini-Batch Loading

Images are loaded in mini-batches of 64 samples using PyTorch's `DataLoader`. Mini-batch gradient descent is used during training rather than processing the entire dataset at once because:

1. It makes gradient updates more frequent, which speeds up learning.
2. The noise introduced by sampling different batches helps the optimizer escape poor local minima.
3. It makes training feasible on hardware with limited memory.

The training DataLoader uses `shuffle=True` so that the order of samples is randomized at every epoch, preventing the model from learning any unintended ordering patterns in the data.

---

## Summary

The preprocessing pipeline ensures that all images are in a consistent, normalized format before being passed to the model. The combination of resizing, normalization, and data augmentation directly addresses key challenges of the GTSRB dataset: variable image sizes, diverse capture conditions, and a limited number of samples per class for rare sign types. These choices follow established best practices in deep learning and are grounded in the theoretical foundations of gradient-based optimization and generalization.
