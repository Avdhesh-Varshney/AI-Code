# 🧪 Long Short-Term Memory (LSTM)

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*eEIAtVm41hnA7Sb9O3z1xg.png" />
</div>

## 🎯 Objective
Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Networks (RNNs) designed to overcome the vanishing gradient problem. They efficiently process sequential data by maintaining long-range dependencies, making them suitable for tasks such as speech recognition, text generation, and time-series forecasting.

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

## 🏩 LSTM Architecture
- LSTMs use **memory cells** and **gates** to regulate the flow of information.
- The key components of an LSTM unit:
  - **Forget Gate**: Decides what information to discard.
  - **Input Gate**: Determines what new information to store.
  - **Cell State**: Maintains long-term dependencies.
  - **Output Gate**: Controls the final output of the cell.
- The update equations are:
$$
f_t = \sigma(W_f \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_f)
$$

$$
i_t = \sigma(W_i \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_C)
$$

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

$$
o_t = \sigma(W_o \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_o)
$$

$$
h_t = o_t \cdot \tanh(C_t)
$$


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

# Define LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Model Training
input_size = 10  # Number of input features
hidden_size = 20  # Number of hidden neurons
output_size = 1   # Output dimension

model = LSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample Training Loop
for epoch in range(10):
    optimizer.zero_grad()
    inputs = torch.randn(32, 5, input_size)  # (batch_size, seq_length, input_size)
    hidden = (torch.zeros(1, 32, hidden_size), torch.zeros(1, 32, hidden_size))  # Initial hidden state
    outputs, hidden = model(inputs, hidden)
    loss = criterion(outputs, torch.randn(32, output_size))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## 🔍 Understanding the Code
- **Model Definition:**
  - The `LSTM` class defines a simple long short-term memory network with an input layer, an LSTM layer, and a fully connected output layer.
- **Forward Pass:**
  - Takes an input sequence, processes it through the LSTM layer, and generates an output.
- **Training Loop:**
  - Uses randomly generated data for demonstration.
  - Optimizes weights using the Adam optimizer and mean squared error loss.

---

## 🌟 Advantages
- Capable of learning long-term dependencies in sequential data.
- Effective in avoiding the vanishing gradient problem.
- Widely used in NLP, speech recognition, and time-series forecasting.

## ⚠️ Limitations
- Computationally expensive compared to simple RNNs.
- Requires careful tuning of hyperparameters.
- Training can be slow for very long sequences.

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
