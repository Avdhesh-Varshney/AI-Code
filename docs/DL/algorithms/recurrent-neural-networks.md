# Convolutional Neural Networks

<div align="center">
  <img src="https://t3.ftcdn.net/jpg/02/61/57/66/360_F_261576629_qbzv83cBaYxMjBCTtY85cHyTK2GFRvk7.jpg" />
</div>

## Overview

Convolutional Neural Networks (CNNs) are a type of deep learning algorithm specifically designed for processing structured grid data such as images. They are widely used in computer vision tasks like image classification, object detection, and image segmentation.

---

## How CNNs Work

### 1. **Architecture**

CNNs are composed of the following layers:

- **Convolutional Layers**: Extract spatial features from the input data.
- **Pooling Layers**: Reduce the spatial dimensions of feature maps to lower computational costs.
- **Fully Connected Layers**: Perform high-level reasoning for final predictions.

### 2. **Key Concepts**

- **Filters (Kernels)**: Small matrices that slide over the input to extract features.
- **Strides**: Step size of the filter movement.
- **Padding**: Adding borders to the input for better filter coverage.
- **Activation Functions**: Introduce non-linearity (e.g., ReLU).

---

## CNN Algorithms

### 1. **LeNet**

- **Proposed By**: Yann LeCun (1998)
- **Use Case**: Handwritten digit recognition (e.g., MNIST dataset).
- **Architecture**:
  - Input → Convolution → Pooling → Convolution → Pooling → Fully Connected → Output

### 2. **AlexNet**

- **Proposed By**: Alex Krizhevsky (2012)
- **Use Case**: ImageNet classification challenge.
- **Key Features**:
  - Uses ReLU for activation.
  - Includes dropout to prevent overfitting.
  - Designed for GPUs for faster computation.

### 3. **VGGNet**

- **Proposed By**: Visual Geometry Group (2014)
- **Use Case**: Image classification and transfer learning.
- **Key Features**:
  - Uses small 3x3 filters.
  - Depth of the network increases (e.g., VGG-16, VGG-19).

### 4. **ResNet**

- **Proposed By**: Kaiming He et al. (2015)
- **Use Case**: Solving vanishing gradient problems in deep networks.
- **Key Features**:
  - Introduces residual blocks with skip connections.
  - Enables training of very deep networks (e.g., ResNet-50, ResNet-101).

### 5. **MobileNet**

- **Proposed By**: Google (2017)
- **Use Case**: Mobile and embedded vision applications.
- **Key Features**:
  - Utilizes depthwise separable convolutions.
  - Lightweight architecture suitable for mobile devices.

---

## Code Example: Implementing a Simple CNN

Here’s a Python example of a CNN using **TensorFlow/Keras**:

- **Sequential:** Used to stack layers to create a neural network model.
- **Conv2D:** Implements the convolutional layers to extract features from input images.
- **MaxPooling2D:** Reduces the size of feature maps while retaining important features.
- **Flatten:** Converts 2D feature maps into a 1D vector to pass into fully connected layers.
- **Dense:** Implements fully connected (dense) layers, responsible for decision-making.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Replace 10 with the number of classes in your dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
```

---

# Visualizations

- **Filters and Feature Maps:** Visualizing how the CNN learns features from images.
- **Training Metrics:** Plotting accuracy and loss during training.

```python
import matplotlib.pyplot as plt

# Example: Visualizing accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
```
