# Ridge Regression

This module contains an implementation of Ridge Regression, a linear regression variant that includes regularization to prevent overfitting.

## Overview

Ridge Regression is a linear regression technique with an added regularization term to handle multicollinearity and prevent the model from becoming too complex.

## Parameters

- `alpha`: Regularization strength. A higher alpha increases the penalty for large coefficients.

## Scratch Code 

- ridge_regression.py file 

```py
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Constructor for the Ridge Regression class.

        Parameters:
        - alpha: Regularization strength. Higher values specify stronger regularization.
        """
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        """
        Fit the Ridge Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Add a column of ones to the input features for the bias term
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Compute the closed-form solution for Ridge Regression
        identity_matrix = np.identity(X_bias.shape[1])
        self.weights = np.linalg.inv(X_bias.T @ X_bias + self.alpha * identity_matrix) @ X_bias.T @ y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        # Add a column of ones to the input features for the bias term
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Make predictions using the learned weights
        predictions = X_bias @ self.weights

        return predictions
```

- ridge_regression_test.py file 

```py
import numpy as np
import unittest
from RidgeRegression import RidgeRegression  # Assuming your RidgeRegression class is in a separate file

class TestRidgeRegression(unittest.TestCase):
    def test_fit_predict(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        X_train = np.random.rand(100, 2)
        y_train = 3 * X_train[:, 0] + 5 * X_train[:, 1] + 2 + 0.1 * np.random.randn(100)
        X_test = np.random.rand(20, 2)

        # Create a Ridge Regression model
        ridge_model = RidgeRegression(alpha=0.1)

        # Fit the model to training data
        ridge_model.fit(X_train, y_train)

        # Make predictions on test data
        predictions = ridge_model.predict(X_test)

        # Ensure the predictions have the correct shape
        self.assertEqual(predictions.shape, (20,))

    def test_invalid_alpha(self):
        # Check if an exception is raised for an invalid alpha value
        with self.assertRaises(ValueError):
            RidgeRegression(alpha=-1)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
```
