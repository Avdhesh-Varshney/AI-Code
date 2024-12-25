# Polynomial Regression

This module contains an implementation of Polynomial Regression, an extension of Linear Regression that models the relationship between the independent variable and the dependent variable as a polynomial.

## Parameters

- `degree`: Degree of the polynomial.
- `learning_rate`: The step size for gradient descent.
- `n_iterations`: The number of iterations for gradient descent.

## Scratch Code 

- polynomial_regression.py file 

```py
import numpy as np

# Polynomial regression implementation
class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the PolynomialRegression class.

        Parameters:
        - degree: Degree of the polynomial.
        - learning_rate: The step size for gradient descent.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _polynomial_features(self, X):
        """
        Create polynomial features up to the specified degree.

        Parameters:
        - X: Input features (numpy array).

        Returns:
        - Polynomial features (numpy array).
        """
        return np.column_stack([X ** i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fit the polynomial regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        X_poly = self._polynomial_features(X)
        self.weights = np.zeros((X_poly.shape[1], 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X_poly, self.weights) + self.bias
            errors = predictions - y

            self.weights -= self.learning_rate * (1 / len(X_poly)) * np.dot(X_poly.T, errors)
            self.bias -= self.learning_rate * (1 / len(X_poly)) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
```

- polynomial_regression_test.py file 

```py
import unittest
import numpy as np
from PolynomialRegression import PolynomialFeatures

class TestPolynomialRegression(unittest.TestCase):

    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

    def test_fit_predict(self):
        # Test the fit and predict methods
        poly_model = PolynomialFeatures(degree=2)
        poly_model.fit(self.X_train, self.y_train)
        
        # Create test data
        X_test = np.array([[1.5], [2.0]])
        
        # Make predictions
        predictions = poly_model.predict(X_test)
        
        # Assert that the predictions are NumPy arrays
        self.assertTrue(isinstance(predictions, np.ndarray))
        
        # Assert that the shape of predictions is as expected
        self.assertEqual(predictions.shape, (X_test.shape[0], 1))

if __name__ == '__main__':
    unittest.main()
```
