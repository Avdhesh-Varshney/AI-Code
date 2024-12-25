# Gradient Boosting Regression

This module contains an implementation of Gradient Boosting Regression, an ensemble learning method that combines multiple weak learners (typically decision trees) to create a more robust and accurate model for predicting continuous outcomes based on input features.

## Parameters

- `n_estimators`: Number of boosting stages (trees) to be run.
- `learning_rate`: Step size shrinkage to prevent overfitting.
- `max_depth`: Maximum depth of each decision tree.

## Scratch Code 

- gradient_boosting_regression.py file 

```py
import numpy as np

class GradientBoostingRegression:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Constructor for the GradientBoostingRegression class.

        Parameters:
        - n_estimators: Number of trees in the ensemble.
        - learning_rate: Step size for each tree's contribution.
        - max_depth: Maximum depth of each decision tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Fit the gradient boosting regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize predictions with the mean of the target values
        predictions = np.mean(y) * np.ones_like(y)

        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions

            # Fit a decision tree to the residuals
            tree = self._fit_tree(X, residuals, depth=0)
            
            # Update predictions using the tree's contribution scaled by the learning rate
            predictions += self.learning_rate * self._predict_tree(X, tree)

            # Save the tree in the ensemble
            self.trees.append(tree)

    def _fit_tree(self, X, y, depth):
        """
        Fit a decision tree to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - depth: Current depth of the tree.

        Returns:
        - Tree structure (dictionary).
        """
        if depth == self.max_depth:
            # If the maximum depth is reached, return the mean of the target values
            return np.mean(y)

        # Find the best split point
        feature_index, threshold = self._find_best_split(X, y)

        if feature_index is None:
            # If no split improves the purity, return the mean of the target values
            return np.mean(y)

        # Split the data
        mask = X[:, feature_index] < threshold
        left_tree = self._fit_tree(X[mask], y[mask], depth + 1)
        right_tree = self._fit_tree(X[~mask], y[~mask], depth + 1)

        # Return the tree structure
        return {'feature_index': feature_index, 'threshold': threshold,
                'left_tree': left_tree, 'right_tree': right_tree}

    def _find_best_split(self, X, y):
        """
        Find the best split point for a decision tree.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).

        Returns:
        - Best feature index and threshold for the split.
        """
        m, n = X.shape
        if m <= 1:
            return None, None  # No split is possible

        # Calculate the initial impurity
        initial_impurity = self._calculate_impurity(y)

        # Initialize variables to store the best split parameters
        best_feature_index, best_threshold, best_impurity_reduction = None, None, 0

        for feature_index in range(n):
            # Sort the feature values and corresponding target values
            sorted_indices = np.argsort(X[:, feature_index])
            sorted_X = X[sorted_indices, feature_index]
            sorted_y = y[sorted_indices]

            # Initialize variables to keep track of impurity and counts for the left and right nodes
            left_impurity, left_count = 0, 0
            right_impurity, right_count = initial_impurity, m

            for i in range(1, m):
                # Update impurity and counts for the left and right nodes
                value = sorted_X[i]
                left_impurity += (i / m) * self._calculate_impurity(sorted_y[i-1:i+1])
                left_count += 1
                right_impurity -= ((i-1) / m) * self._calculate_impurity(sorted_y[i-1:i+1])
                right_count -= 1

                # Calculate impurity reduction
                impurity_reduction = initial_impurity - (left_count / m * left_impurity + right_count / m * right_impurity)

                # Check if this is the best split so far
                if impurity_reduction > best_impurity_reduction:
                    best_feature_index = feature_index
                    best_threshold = value
                    best_impurity_reduction = impurity_reduction

        return best_feature_index, best_threshold

    def _calculate_impurity(self, y):
        """
        Calculate the impurity of a node.

        Parameters:
        - y: Target values (numpy array).

        Returns:
        - Impurity.
        """
        # For regression, impurity is the variance of the target values
        return np.var(y)

    def _predict_tree(self, X, tree):
        """
        Make predictions using a decision tree.

        Parameters:
        - X: Input features (numpy array).
        - tree: Tree structure (dictionary).

        Returns:
        - Predicted values (numpy array).
        """
        if 'feature_index' not in tree:
            # If the node is a leaf, return the constant value
            return tree
        else:
            # Recursively traverse the tree
            mask = X[:, tree['feature_index']] < tree['threshold']
            return np.where(mask, self._predict_tree(X, tree['left_tree']), self._predict_tree(X, tree['right_tree']))

    def predict(self, X):
        """
        Make predictions on new data using the Gradient Boosting Regression.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        predictions = np.sum(self.learning_rate * self._predict_tree(X, tree) for tree in self.trees)
        return predictions
```

- gradient_boosting_regression_test.py file 

```py
import unittest
import numpy as np
from GradientBoostingRegressor import GradientBoostingRegression

class TestGradientBoostingRegressor(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = 2 * self.X_train[:, 0] + 3 * self.X_train[:, 1] + np.random.normal(0, 0.1, 100)

        self.X_test = np.random.rand(10, 2)

    def test_fit_predict(self):
        # Test if the model can be fitted and predictions are made
        gbr_model = GradientBoostingRegression(n_estimators=5, learning_rate=0.1, max_depth=3)
        gbr_model.fit(self.X_train, self.y_train)

        # Ensure predictions are made without errors
        predictions = gbr_model.predict(self.X_test)

        # Add your specific assertions based on the expected behavior of your model
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (10,))

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
```
