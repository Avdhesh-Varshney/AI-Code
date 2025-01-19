# Linear Regression

This module contains an implementation of the Linear Regression algorithm, a fundamental technique in machine learning for predicting a continuous outcome based on input features.

## Parameters

- `learning_rate`: The step size for gradient descent.
- `n_iterations`: The number of iterations for gradient descent.

## Scratch Code 

- linear_regression.py file 

```py
import numpy as np

# Linear regression implementation
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the LinearRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - n_iterations: The number of iterations for gradient descent.
        - n_iterations: n_epochs.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize weights and bias
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Compute predictions
            predictions = np.dot(X, self.weights) + self.bias

            # Calculate errors
            errors = predictions - y

            # Update weights and bias
            self.weights -= self.learning_rate * (1 / len(X)) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1 / len(X)) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        return np.dot(X, self.weights) + self.bias
```

- linear_regression_test.py file 

```py
import unittest
import numpy as np
from LinearRegression import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Set up some common data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

        self.X_test = 2 * np.random.rand(20, 1)
        self.y_test = 4 + 3 * self.X_test + np.random.randn(20, 1)

    def test_fit_predict(self):
        # Test the fit and predict methods

        # Create a LinearRegression model
        lr_model = LinearRegression()

        # Fit the model to the training data
        lr_model.fit(self.X_train, self.y_train)

        # Make predictions on the test data
        predictions = lr_model.predict(self.X_test)

        # Check that the predictions are of the correct shape
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_predict_with_unfitted_model(self):
        # Test predicting with an unfitted model

        # Create a LinearRegression model (not fitted)
        lr_model = LinearRegression()

        # Attempt to make predictions without fitting the model
        with self.assertRaises(ValueError):
            _ = lr_model.predict(self.X_test)

if __name__ == '__main__':
    unittest.main()
```
