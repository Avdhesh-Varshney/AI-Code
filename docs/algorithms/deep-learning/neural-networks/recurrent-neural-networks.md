# Recurrent Neural Networks (RNN)

---

## **What is a Recurrent Neural Network (RNN)?**

A **Recurrent Neural Network (RNN)** is a type of artificial neural network designed for modeling **sequential data**. Unlike traditional feedforward networks, RNNs have the capability to remember information from previous time steps, making them well-suited for tasks involving temporal or sequential relationships.

### **Key Characteristics of RNN**:
- **Sequential Processing**: Processes inputs sequentially, one step at a time.
- **Memory Capability**: Uses hidden states to store information about previous steps.
- **Shared Weights**: The same weights are applied across all time steps, reducing complexity.

---

## **Architecture of RNN**

### **Components of RNN**:
1. **Input Layer**:
   - Accepts sequential input data (e.g., time-series data, text, or audio signals).

2. **Hidden Layer with Recurrence**:
   - Maintains a **hidden state** \( h_t \), which is updated at each time step based on the input and the previous hidden state.
   - Formula:  
     \[
     h_t = f(W_h \cdot h_{t-1} + W_x \cdot x_t + b)
     \]
     Where:
     - \( h_t \): Current hidden state.
     - \( h_{t-1} \): Previous hidden state.
     - \( x_t \): Input at time step \( t \).
     - \( W_h, W_x \): Weight matrices.
     - \( b \): Bias.
     - \( f \): Activation function (e.g., tanh or ReLU).

3. **Output Layer**:
   - Produces output based on the current hidden state.
   - Formula:  
     \[
     y_t = g(W_y \cdot h_t + c)
     \]
     Where:
     - \( y_t \): Output at time step \( t \).
     - \( W_y \): Output weight matrix.
     - \( c \): Output bias.
     - \( g \): Activation function (e.g., softmax or sigmoid).

---

## **Types of RNNs**

### **1. Vanilla RNN**:
- Standard RNN that processes sequential data using the hidden state.
- Struggles with long-term dependencies due to **vanishing gradient problems**.

### **2. Long Short-Term Memory (LSTM)**:
- A specialized type of RNN that can learn long-term dependencies by using **gates** to control the flow of information.
- Components:
  - **Forget Gate**: Decides what to forget.
  - **Input Gate**: Decides what to store.
  - **Output Gate**: Controls the output.

### **3. Gated Recurrent Unit (GRU)**:
- A simplified version of LSTM that combines the forget and input gates into a single **update gate**.

---

## **Applications of RNN**

### **1. Natural Language Processing (NLP)**:
- Text generation (e.g., predictive typing, chatbots).
- Sentiment analysis.
- Language translation.

### **2. Time-Series Analysis**:
- Stock price prediction.
- Weather forecasting.
- Energy demand forecasting.

### **3. Speech and Audio Processing**:
- Speech-to-text transcription.
- Music generation.

### **4. Video Analysis**:
- Video captioning.
- Action recognition.

---

## **Advantages of RNN**
- Can handle sequential and time-dependent data.
- Shared weights reduce model complexity.
- Effective for tasks with context dependencies, such as language modeling.

---

## **Limitations of RNN**
- **Vanishing Gradient Problem**:
  - Makes it difficult to learn long-term dependencies.
- Computationally expensive for long sequences.
- Struggles with parallelization compared to other architectures like CNNs.

---

## **Code Example: Implementing RNN Using Keras**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Build the RNN model
model = Sequential([
    SimpleRNN(128, activation='tanh', input_shape=(10, 1)),  # 10 timesteps, 1 feature
    Dense(1, activation='sigmoid')                          # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
```

---

# **Applications in Real-world Projects**
* Use RNN for tasks involving sequential data where past information impacts the future.
* Prefer LSTM or GRU over vanilla RNN for learning long-term dependencies.
