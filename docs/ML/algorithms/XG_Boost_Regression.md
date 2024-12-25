# XG Boost Regression 

This module contains an implementation of the XGBoost Regressor, a popular ensemble learning algorithm that combines the predictions from multiple decision trees to create a more robust and accurate model for regression tasks.

## Parameters

- `n_estimators`: Number of boosting rounds (trees).
- `learning_rate`: Step size shrinkage to prevent overfitting.
- `max_depth`: Maximum depth of each tree.
- `gamma`: Minimum loss reduction required to make a further partition.

## Scratch Code 

- x_g_boost_regression.py file 

```py
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class XGBoostRegressor:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, gamma=0):
        """
        Constructor for the XGBoostRegressor class.

        Parameters:
        - n_estimators: Number of boosting rounds (trees).
        - learning_rate: Step size shrinkage to prevent overfitting.
        - max_depth: Maximum depth of each tree.
        - gamma: Minimum loss reduction required to make a further partition.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
        self.trees = []

    def fit(self, X, y):
        """
        Fit the XGBoost model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize residuals
        residuals = np.copy(y)

        for _ in range(self.n_estimators):
            # Fit a weak learner (decision tree) to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.gamma)
            tree.fit(X, residuals)

            # Compute predictions from the weak learner
            predictions = tree.predict(X)

            # Update residuals with the weighted sum of previous residuals and predictions
            residuals -= self.learning_rate * predictions

            # Store the tree in the list
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        # Initialize predictions with zeros
        predictions = np.zeros(X.shape[0])

        # Make predictions using each tree and update the overall prediction
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
```

- x_g_boost_regression_test.py file 

```py
import unittest
import numpy as np
from XGBoostRegressor import XGBoostRegressor

class TestXGBoostRegressor(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100)
        self.X_test = np.random.rand(20, 5)

    def test_fit_predict(self):
        # Test the fit and predict methods
        xgb_model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, gamma=0.1)
        xgb_model.fit(self.X_train, self.y_train)
        predictions = xgb_model.predict(self.X_test)

        # Ensure predictions have the correct shape
        self.assertEqual(predictions.shape, (20,))

    def test_invalid_parameters(self):
        # Test invalid parameter values
        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=-1, learning_rate=0.1, max_depth=3, gamma=0.1)

        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=50, learning_rate=-0.1, max_depth=3, gamma=0.1)

        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=-3, gamma=0.1)

    def test_invalid_fit(self):
        # Test fitting with mismatched X_train and y_train shapes
        xgb_model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, gamma=0.1)
        with self.assertRaises(ValueError):
            xgb_model.fit(self.X_train, np.random.rand(50))

if __name__ == '__main__':
    unittest.main()
```
