# Neural Network Regression

This module contains an implementation of Neural Network Regression, a powerful algorithm for predicting continuous outcomes based on input features.

## Parameters

- `input_size`: Number of features in the input data.
- `hidden_size`: Number of neurons in the hidden layer.
- `output_size`: Number of output neurons.
- `learning_rate`: Step size for updating weights during training.
- `n_iterations`: Number of iterations for training the neural network.

## Scratch Code 

- neural_network_regression.py file 

```py
import numpy as np

class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the NeuralNetworkRegression class.

        Parameters:
        - input_size: Number of input features.
        - hidden_size: Number of neurons in the hidden layer.
        - output_size: Number of output neurons.
        - learning_rate: Step size for gradient descent.
        - n_iterations: Number of iterations for gradient descent.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    def fit(self, X, y):
        """
        Fit the Neural Network model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        for _ in range(self.n_iterations):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            predicted_output = self.sigmoid(output_layer_input)

            # Backpropagation
            error = y - predicted_output
            output_delta = error * self.sigmoid_derivative(predicted_output)

            hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * self.learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

            self.weights_input_hidden += X.T.dot(hidden_layer_delta) * self.learning_rate
            self.bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)

        return predicted_output
```

- neural_network_regression_test.py file 

```py
import numpy as np
import unittest
from NeuralNetworkRegression import NeuralNetworkRegression

class TestNeuralNetworkRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 3)
        self.y_train = np.random.rand(100, 1)

        self.X_test = np.random.rand(10, 3)

    def test_fit_predict(self):
        # Initialize and fit the model
        model = NeuralNetworkRegression(input_size=3, hidden_size=4, output_size=1, learning_rate=0.01, n_iterations=1000)
        model.fit(self.X_train, self.y_train)

        # Ensure predictions have the correct shape
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape, (10, 1))

if __name__ == '__main__':
    unittest.main()
```
