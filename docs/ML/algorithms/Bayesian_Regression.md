# Bayesian Regression 

This module contains an implementation of Bayesian Regression, a probabilistic approach to linear regression that provides uncertainty estimates for predictions.

## Overview

Bayesian Regression is an extension of traditional linear regression that models the distribution of coefficients, allowing for uncertainty in the model parameters. It's particularly useful when dealing with limited data and provides a full probability distribution over the possible values of the regression coefficients.

## Parameters

- `alpha`: Prior precision for the coefficients.
- `beta`: Precision of the noise in the observations.

## Scratch Code 

- bayesian_regression.py file 

```py
import numpy as np

class BayesianRegression:
    def __init__(self, alpha=1, beta=1):
        """
        Constructor for the BayesianRegression class.

        Parameters:
        - alpha: Prior precision.
        - beta: Noise precision.
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def fit(self, X, y):
        """
        Fit the Bayesian Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Add a bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute posterior precision and mean
        self.w_precision = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        self.w_mean = self.beta * np.linalg.solve(self.w_precision, X.T @ y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        # Add a bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute predicted mean
        y_pred = X @ self.w_mean

        return y_pred
```

- bayesian_regression_test.py file 

```py
import unittest
import numpy as np
from BayesianRegression import BayesianRegression

class TestBayesianRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

        self.X_test = 2 * np.random.rand(20, 1)

    def test_fit_predict(self):
        blr = BayesianRegression()
        blr.fit(self.X_train, self.y_train)
        y_pred = blr.predict(self.X_test)

        self.assertTrue(y_pred.shape == (20, 1))

if __name__ == '__main__':
    unittest.main()
```
