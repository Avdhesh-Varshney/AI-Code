# Recurrent Neural Networks (RNNs)

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/2560px-Recurrent_neural_network_unfold.svg.png" />
</div>

## Overview
Recurrent Neural Networks (RNNs) are a class of neural networks designed to model sequential data. They are particularly effective for tasks involving time series, natural language processing, and any application where context is crucial.

---

## How RNNs Work

### 1. **Architecture**
RNNs process sequential data by maintaining a hidden state that captures information about previous inputs. Key components include:
- **Input Layer**: Accepts input data, often in a time-series format.
- **Recurrent Layer**: Includes connections between neurons that form loops, enabling memory of past computations.
- **Output Layer**: Produces predictions based on the current hidden state.

### 2. **Key Concepts**
- **Hidden State**: Acts as memory, retaining context from prior inputs.
- **Weight Sharing**: Parameters are shared across time steps, reducing the complexity of the model.
- **Backpropagation Through Time (BPTT)**: A specialized training process to compute gradients over sequences.

---

## RNN Variants

### 1. **Basic RNN**
- **Use Case**: Simplest form of RNN, but struggles with long-term dependencies due to vanishing gradients.
- **Architecture**:
  - Input → Hidden Layer → Output
  - Hidden state is updated at each time step.

### 2. **Long Short-Term Memory (LSTM)**
- **Proposed By**: Sepp Hochreiter and Jürgen Schmidhuber (1997)
- **Use Case**: Overcomes vanishing gradient issues in basic RNNs.
- **Key Features**:
  - Incorporates **forget gates**, **input gates**, and **output gates**.
  - Capable of learning long-term dependencies.

### 3. **Gated Recurrent Unit (GRU)**
- **Proposed By**: Kyunghyun Cho et al. (2014)
- **Use Case**: Simplifies LSTM by combining forget and input gates.
- **Key Features**:
  - Fewer parameters compared to LSTM.
  - Suitable for smaller datasets or faster computation.

### 4. **Bidirectional RNN**
- **Use Case**: Utilizes both past and future context for prediction.
- **Key Features**:
  - Processes input sequences in both forward and backward directions.

---

## Code Example: Implementing a Simple RNN

Here’s a Python example of a basic RNN using **TensorFlow/Keras**:

* **SimpleRNN:** Implements the recurrent layer.
* **LSTM/GRU:** Alternate layers that can replace SimpleRNN for enhanced performance.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

# Build the RNN
model = Sequential([
    Embedding(input_dim=1000, output_dim=64),  # Embedding for text data (vocab size: 1000)
    SimpleRNN(128, activation='relu', return_sequences=False),
    Dense(10, activation='softmax')  # Replace 10 with the number of classes in your dataset
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
```

---

# Visualizations
* **Hidden States**: Visualizing how the hidden states evolve over time.
* **Training Metrics**: Plotting accuracy and loss during training.

```python
import matplotlib.pyplot as plt

# Example: Visualizing accuracy and loss
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
```
