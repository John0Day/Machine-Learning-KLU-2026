# Baseline Model and Training

## Model Architecture

The baseline model is a convolutional neural network (CNN) designed for multi-class image classification. The architecture follows the general structure of early convolutional networks and consists of two main components: a feature extractor and a classifier.

### Feature Extractor

The feature extractor consists of three convolutional blocks. Each block applies the following sequence of operations:

1. **Convolutional layer** – applies a set of learnable filters to the input. Each filter slides across the image and computes a dot product between its weights and the corresponding local region of the input, producing a feature map that highlights patterns such as edges, corners, or textures.
2. **Batch Normalization** – normalizes the output of the convolutional layer across the batch dimension. This stabilizes training by reducing internal covariate shift and allows for higher learning rates.
3. **ReLU activation** – applies the Rectified Linear Unit function `φ(z) = max(0, z)`, which introduces non-linearity while avoiding the vanishing gradient problem associated with sigmoid activations.
4. **Max Pooling** – reduces the spatial dimensions by a factor of 2 by selecting the maximum value within each 2×2 region. This makes the representation more compact and introduces a degree of spatial invariance.

The three blocks progressively increase the number of filters (32 → 64 → 128) while halving the spatial resolution at each step (32×32 → 16×16 → 8×8 → 4×4). This design allows the network to learn increasingly abstract features at each layer.

### Classifier

After the feature extractor, the output is flattened into a one-dimensional vector of size 128 × 4 × 4 = 2048. This is passed through two fully connected layers:

- **Linear(2048 → 256) + ReLU** – learns a compact representation from the extracted features.
- **Dropout(0.5)** – randomly sets 50% of neurons to zero during training. This acts as a regularization technique, preventing the network from relying too heavily on specific neurons and thereby reducing overfitting.
- **Linear(256 → 43)** – produces one raw score (logit) per class. The final class prediction is the index with the highest logit.

Note that no Softmax activation is applied at the output layer. Instead, Softmax is implicitly computed inside the Cross-Entropy loss function during training, which is numerically more stable.

---

## Loss Function

Cross-Entropy loss is used as the training objective. For a single sample with true class label `y` and predicted class probabilities `p`, it is defined as:

```
L = -log(p_y)
```

Cross-Entropy is the standard choice for multi-class classification tasks because it directly penalizes the model for assigning low probability to the correct class. Compared to mean squared error, it produces stronger gradients when the predicted probability is far from the true label, leading to faster convergence.

---

## Optimizer

The Adam optimizer is used to update the model weights. Adam extends standard gradient descent by maintaining adaptive learning rates for each parameter, computed from the first and second moments of the gradients. This makes it robust to different gradient magnitudes across layers and generally converges faster than plain SGD without requiring careful manual tuning of the learning rate.

A learning rate of 0.001 is used as the initial value. A `ReduceLROnPlateau` scheduler reduces the learning rate by a factor of 0.5 whenever the validation loss does not improve for 3 consecutive epochs, allowing finer optimization in later stages of training.

---

## Training Procedure

The model is trained using mini-batch gradient descent with a batch size of 64. For each mini-batch, the following steps are performed:

1. **Forward pass** – compute the model output for the current batch.
2. **Loss computation** – evaluate the Cross-Entropy loss between the output and the true labels.
3. **Backward pass** – compute gradients of the loss with respect to all model parameters using backpropagation.
4. **Weight update** – update the parameters in the direction opposite to the gradient, scaled by the learning rate.

Training runs for a maximum of 30 epochs. After each epoch, the model is evaluated on the validation set. The model weights that achieve the highest validation accuracy are saved as the best model.

---

## Early Stopping

To prevent overfitting, early stopping is applied with a patience of 5 epochs. If the validation accuracy does not improve for 5 consecutive epochs, training is terminated early. This avoids wasting computation on epochs where the model is no longer generalizing better, and reduces the risk of the model memorizing training data rather than learning general patterns.

---

## Summary

| Component | Choice | Reason |
|---|---|---|
| Architecture | 3-block CNN | Sufficient depth for 32×32 input |
| Activation | ReLU | Avoids vanishing gradients |
| Regularization | Dropout + BatchNorm | Reduces overfitting |
| Loss | Cross-Entropy | Standard for multi-class classification |
| Optimizer | Adam | Adaptive learning rate, fast convergence |
| Early Stopping | Patience = 5 | Prevents overfitting, saves compute |
