# Lasso Regression

This module contains an implementation of Lasso Regression, a linear regression technique with L1 regularization.

## Overview

Lasso Regression is a regression algorithm that adds a penalty term based on the absolute values of the coefficients. This penalty term helps in feature selection by driving some of the coefficients to exactly zero, effectively ignoring certain features.

## Parameters

- `learning_rate`: The step size for gradient descent.
- `lambda_param`: Regularization strength (L1 penalty).
- `n_iterations`: The number of iterations for gradient descent.

## Scratch Code 

- lasso_regression.py file 

```py
import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        """
        Constructor for the LassoRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - lambda_param: Regularization strength.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Lasso Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent
        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = y - predictions

            # Update weights and bias
            self.weights += self.learning_rate * (1/num_samples) * (np.dot(X.T, errors) - self.lambda_param * np.sign(self.weights))
            self.bias += self.learning_rate * (1/num_samples) * np.sum(errors)

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

- lasso_regression_test.py file 

```py
import unittest
import numpy as np
from LassoRegression import LassoRegression

class TestLassoRegression(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = 3 * self.X_train[:, 0] + 2 * self.X_train[:, 1] + np.random.randn(100)

        self.X_test = np.random.rand(10, 2)

    def test_fit_predict(self):
        # Test the fit and predict methods
        model = LassoRegression(learning_rate=0.01, lambda_param=0.1, n_iterations=1000)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        # Ensure predictions are of the correct shape
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
```
