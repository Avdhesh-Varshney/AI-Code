# Neural Network Regression

This module contains an implementation of Neural Network Regression, a powerful algorithm for predicting continuous outcomes based on input features.

## Usage

To use Neural Network Regression, follow these steps:

1. Import the `NeuralNetworkRegression` class.
2. Create an instance of the class, specifying parameters such as the input size, hidden layer size, output size, learning rate, and number of iterations.
3. Fit the model to your training data using the `fit` method.
4. Make predictions using the `predict` method.

Example:

```python
from NeuralNetworkRegression import NeuralNetworkRegression

nn_model = NeuralNetworkRegression(input_size=3, hidden_size=4, output_size=1, learning_rate=0.01, n_iterations=1000)
nn_model.fit(X_train, y_train)
predictions = nn_model.predict(X_test)
```

## Parameters

- `input_size`: Number of features in the input data.
- `hidden_size`: Number of neurons in the hidden layer.
- `output_size`: Number of output neurons.
- `learning_rate`: Step size for updating weights during training.
- `n_iterations`: Number of iterations for training the neural network.

## Installation

To use this module, make sure you have the required dependencies installed:

```bash
pip install numpy
```

## Coded By 

[Avdhesh Varshney](https://github.com/Avdhesh-Varshney)

### Happy Coding 👦
