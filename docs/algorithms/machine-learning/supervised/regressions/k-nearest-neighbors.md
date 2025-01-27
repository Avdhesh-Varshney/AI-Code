# K Nearest Neighbors Regression

This module contains an implementation of K-Nearest Neighbors Regression, a simple yet effective algorithm for predicting continuous outcomes based on input features.

## Parameters

- `k`: Number of neighbors to consider for prediction.

## Scratch Code 

- k_nearest_neighbors_regression.py file 

```py
import numpy as np

class KNNRegression:
    def __init__(self, k=5):
        """
        Constructor for the KNNRegression class.

        Parameters:
        - k: Number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        predictions = []
        for x in X:
            # Calculate Euclidean distances between the input point and all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k-nearest neighbors
            indices = np.argsort(distances)[:self.k]

            # Average the target values of k-nearest neighbors
            predicted_value = np.mean(self.y_train[indices])
            predictions.append(predicted_value)

        return np.array(predictions)
```

- k_nearest_neighbors_regression_test.py file 

```py
import unittest
import numpy as np
from KNearestNeighborsRegression import KNNRegression

class TestKNNRegression(unittest.TestCase):

    def test_knn_regression(self):
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

        X_test = np.array([[2.5], [5.0], [7.5]])

        # Initialize and fit the KNN Regression model
        knn_model = KNNRegression(k=3)
        knn_model.fit(X_train, y_train)

        # Test predictions
        predictions = knn_model.predict(X_test)
        expected_predictions = [2 * 2.5, 2 * 5.0, 2 * 7.5]  # Assuming a linear relationship

        # Check if predictions are close to the expected values
        np.testing.assert_allclose(predictions, expected_predictions, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
```
