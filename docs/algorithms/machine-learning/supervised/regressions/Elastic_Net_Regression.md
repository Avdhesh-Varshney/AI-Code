# Elastic Net Regression

This module contains an implementation of Elastic Net Regression, a powerful linear regression technique that combines both L1 (Lasso) and L2 (Ridge) regularization. Elastic Net is particularly useful when dealing with high-dimensional datasets and can effectively handle correlated features.

## Parameters

- `alpha`: The regularization strength. A positive float value.
- `l1_ratio`: The ratio of L1 regularization to L2 regularization. Should be between 0 and 1.
- `max_iter`: The maximum number of iterations to run the optimization algorithm.
- `tol`: The tolerance for the optimization. If the updates are smaller than this value, the optimization will stop.

## Scratch Code 

- elastic_net_regression.py file 

```py
import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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

            gradient_w = (-2 / n_samples) * (X.T.dot(error)) + self.alpha * (self.l1_ratio * np.sign(self.coef_) + (1 - self.l1_ratio) * 2 * self.coef_)
            gradient_b = (-2 / n_samples) * np.sum(error)

            new_coef = self.coef_ - learning_rate * gradient_w
            new_intercept = self.intercept_ - learning_rate * gradient_b

            if np.all(np.abs(new_coef - self.coef_) < self.tol) and np.abs(new_intercept - self.intercept_) < self.tol:
                break

            self.coef_ = new_coef
            self.intercept_ = new_intercept

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
```

- elastic_net_regression_test.py file 

```py
import unittest
import numpy as np
from sklearn.linear_model import ElasticNet
from ElasticNetRegression import ElasticNetRegression

class TestElasticNetRegression(unittest.TestCase):

    def test_elastic_net_regression(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2 

        X_test = np.array([[2.5], [5.0], [7.5]])

        custom_model = ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)

        sklearn_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)

        np.testing.assert_allclose(custom_predictions, sklearn_predictions, rtol=1e-1)

        train_predictions_custom = custom_model.predict(X_train)
        train_predictions_sklearn = sklearn_model.predict(X_train)

        custom_mse = np.mean((y_train - train_predictions_custom) ** 2)
        sklearn_mse = np.mean((y_train - train_predictions_sklearn) ** 2)

        print(f"Custom Model MSE: {custom_mse}")
        print(f"Scikit-learn Model MSE: {sklearn_mse}")

if __name__ == '__main__':
    unittest.main()
```
