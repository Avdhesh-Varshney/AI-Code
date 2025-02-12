# 🧪 Gated Recurrent Unit (GRU)

<div align="center">
    <img src="https://www.researchgate.net/publication/350463426/figure/fig4/AS:1012181290201090@1618334233899/Structure-of-the-gated-recurrent-unit-GRU-recurrent-network.jpg" />
</div>

## 🎯 Objective
Gated Recurrent Units (GRUs) are a variant of Recurrent Neural Networks (RNNs) designed to address the vanishing gradient problem. They improve upon traditional RNNs by using gating mechanisms to regulate information flow, making them efficient for sequential data tasks like speech recognition, text generation, and time-series forecasting.

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

## 🏩 GRU Architecture
- GRUs use **gates** to control information flow without separate memory cells.
- The key components of a GRU unit:
  - **Reset Gate**: Controls how much past information to forget.
  - **Update Gate**: Determines how much of the new information to store.
  - **Hidden State**: Stores relevant sequence information.
- The update equations are:
  $$ r_t = \sigma(W_r [h_{t-1}, x_t] + b_r) $$
  $$ z_t = \sigma(W_z [h_{t-1}, x_t] + b_z) $$
  $$ \tilde{h}_t = \tanh(W_h [r_t * h_{t-1}, x_t] + b_h) $$
  $$ h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t $$

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

# Define GRU Model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Model Training
input_size = 10  # Number of input features
hidden_size = 20  # Number of hidden neurons
output_size = 1   # Output dimension

model = GRU(input_size, hidden_size, output_size)
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
  - The `GRU` class defines a simple Gated Recurrent Unit network with an input layer, a GRU layer, and a fully connected output layer.
- **Forward Pass:**
  - Takes an input sequence, processes it through the GRU layer, and generates an output.
- **Training Loop:**
  - Uses randomly generated data for demonstration.
  - Optimizes weights using the Adam optimizer and mean squared error loss.

---

## 🌟 Advantages
- Requires fewer parameters than LSTMs, making it computationally efficient.
- Captures long-term dependencies while avoiding vanishing gradients.
- Suitable for NLP, speech recognition, and time-series analysis.

## ⚠️ Limitations
- May not perform as well as LSTMs in tasks requiring precise memory retention.
- Still requires careful hyperparameter tuning.
- Training can be slow for very long sequences.

## 🚀 Applications

1. Natural Language Processing (NLP)
- Text prediction
- Sentiment analysis
- Machine translation

2. Time-Series Forecasting
- Stock price prediction
- Weather forecasting
- Healthcare monitoring (e.g., ECG signals)

---
