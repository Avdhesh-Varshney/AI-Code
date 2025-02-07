# 🧪 Recurrent Neural Network (RNN)

<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/512px-Recurrent_neural_network_unfold.svg.png" />
</div>

## 🎯 Objective
Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to process sequential data. Unlike feedforward networks, RNNs have connections that allow information to persist, making them suitable for tasks such as speech recognition, text generation, and time-series forecasting.

## 📚 Prerequisites
- Understanding of basic neural networks and deep learning
- Knowledge of activation functions and backpropagation
- Familiarity with sequence-based data processing
- Libraries: NumPy, TensorFlow, PyTorch

---

## 🧬 Inputs
- A sequence of data points such as text, speech signals, or time-series data.
- Example: A sentence represented as a sequence of word embeddings for NLP tasks.

## 🎎 Outputs
- Predicted sequence values or classifications.
- Example: Next word prediction in a sentence or stock price forecasting.

---

## 🏩 RNN Architecture
- RNNs maintain a **hidden state** that updates with each time step.
- At each step, the hidden state is computed as:
  $$ h_t = f(W_h h_{t-1} + W_x x_t + b) $$
- Variants of RNNs include **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)**, which help mitigate the vanishing gradient problem.

## 🏅 Training Process
- The model is trained using **Backpropagation Through Time (BPTT)**.
- Uses optimizers like **Adam** or **SGD**.
- Typical hyperparameters:
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: 30
  - Loss function: Cross-entropy for classification tasks, MSE for regression tasks.

## 📊 Evaluation Metrics
- Accuracy (for classification)
- Perplexity (for language models)
- Mean Squared Error (MSE) (for regression tasks)
- BLEU Score (for sequence-to-sequence models)

---

## 💻 Code Implementation
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Model Training
input_size = 10  # Number of input features
hidden_size = 20  # Number of hidden neurons
output_size = 1   # Output dimension

model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample Training Loop
for epoch in range(10):
    optimizer.zero_grad()
    inputs = torch.randn(32, 5, input_size)  # (batch_size, seq_length, input_size)
    hidden = torch.zeros(1, 32, hidden_size)  # Initial hidden state
    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs, torch.randn(32, output_size))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## 🔍 Understanding the Code
- **Model Definition:**
  - The `RNN` class defines a simple recurrent neural network with an input layer, a recurrent layer, and a fully connected output layer.
- **Forward Pass:**
  - Takes an input sequence, processes it through the RNN layer, and generates an output.
- **Training Loop:**
  - Uses randomly generated data for demonstration.
  - Optimizes weights using the Adam optimizer and mean squared error loss.

---

## 🌟 Advantages
- Effective for sequential data modeling.
- Capable of handling variable-length inputs.
- Works well for applications like text generation and speech recognition.

## ⚠️ Limitations
- Struggles with long-range dependencies due to vanishing gradients.
- Training can be slow due to sequential computations.
- Alternatives like **LSTMs and GRUs** are preferred for longer sequences.

## 🚀 Applications
### Natural Language Processing (NLP)
- Text prediction
- Sentiment analysis
- Machine translation

### Time-Series Forecasting
- Stock price prediction
- Weather forecasting
- Healthcare monitoring (e.g., ECG signals)

---


