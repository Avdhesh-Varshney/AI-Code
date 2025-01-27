# Huber Regression

This module contains an implementation of Huber Regression, a robust linear regression technique that combines the properties of both least squares and absolute error loss functions. Huber Regression is particularly useful when dealing with datasets that have outliers, as it is less sensitive to outliers compared to standard linear regression.

## Overview

Huber Regression is a regression algorithm that adds a penalty based on the Huber loss function. This loss function is quadratic for small errors and linear for large errors, providing robustness against outliers.

## Parameters

- `alpha`: The regularization strength. A positive float value.
- `epsilon`: The threshold for the Huber loss function. A positive float value.
- `max_iter`: The maximum number of iterations to run the optimization algorithm.
- `tol`: The tolerance for the optimization. If the updates are smaller than this value, the optimization will stop.

## Scratch Code 

- huber_regression.py file 

```py
import numpy as np

class HuberRegression:
    def __init__(self, alpha=1.0, epsilon=1.35, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        learning_rate = 0.01

        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = y - y_pred

            # Compute Huber gradient
            mask = np.abs(error) <= self.epsilon
            gradient_w = (-2 / n_samples) * (X.T.dot(error * mask) + self.epsilon * np.sign(error) * (~mask)) + self.alpha * self.coef_
            gradient_b = (-2 / n_samples) * (np.sum(error * mask) + self.epsilon * np.sign(error) * (~mask))

            new_coef = self.coef_ - learning_rate * gradient_w
            new_intercept = self.intercept_ - learning_rate * gradient_b

            if np.all(np.abs(new_coef - self.coef_) < self.tol) and np.abs(new_intercept - self.intercept_) < self.tol:
                break

            self.coef_ = new_coef
            self.intercept_ = new_intercept

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

- huber_regression_test.py file 

```py
import unittest
import numpy as np
from sklearn.linear_model import HuberRegressor
from HuberRegression import HuberRegression

class TestHuberRegression(unittest.TestCase):

    def test_huber_regression(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2

        X_test = np.array([[2.5], [5.0], [7.5]])

        huber_model = HuberRegression(alpha=1.0, epsilon=1.35)
        huber_model.fit(X_train, y_train)
        huber_predictions = huber_model.predict(X_test)

        sklearn_model = HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=1000, tol=1e-4)
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)

        np.testing.assert_allclose(huber_predictions, sklearn_predictions, rtol=1e-1)

        train_predictions_huber = huber_model.predict(X_train)
        train_predictions_sklearn = sklearn_model.predict(X_train)

        huber_mse = np.mean((y_train - train_predictions_huber) ** 2)
        sklearn_mse = np.mean((y_train - train_predictions_sklearn) ** 2)

        print(f"Huber Model MSE: {huber_mse}")
        print(f"Scikit-learn Model MSE: {sklearn_mse}")

if __name__ == '__main__':
    unittest.main()
```
