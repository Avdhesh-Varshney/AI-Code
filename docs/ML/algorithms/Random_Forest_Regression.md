# Random Forest Regression

This module contains an implementation of Random Forest Regression, an ensemble learning method that combines multiple decision trees to create a more robust and accurate model for predicting continuous outcomes based on input features.

## Parameters

- `n_trees`: Number of trees in the random forest.
- `max_depth`: Maximum depth of each decision tree.
- `max_features`: Maximum number of features to consider for each split.

## Scratch Code 

- random_forest_regression.py file 

```py
import numpy as np

class RandomForestRegression:

    def __init__(self, n_trees=100, max_depth=None, max_features=None):
        """
        Constructor for the RandomForestRegression class.

        Parameters:
        - n_trees: Number of trees in the random forest.
        - max_depth: Maximum depth of each decision tree.
        - max_features: Maximum number of features to consider for each split.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the dataset.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).

        Returns:
        - Bootstrap sample of X and y.
        """
        indices = np.random.choice(len(X), len(X), replace=True)
        return X[indices], y[indices]

    def _build_tree(self, X, y, depth):
        """
        Recursively build a decision tree.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - depth: Current depth of the tree.

        Returns:
        - Node of the decision tree.
        """
        if depth == self.max_depth or np.all(y == y[0]):
            return {'value': np.mean(y)}

        n_features = X.shape[1]
        if self.max_features is None:
            subset_features = np.arange(n_features)
        else:
            subset_features = np.random.choice(n_features, self.max_features, replace=False)

        # Create a random subset of features for this tree
        X_subset = X[:, subset_features]

        # Create a bootstrap sample
        X_bootstrap, y_bootstrap = self._bootstrap_sample(X_subset, y)

        # Find the best split using the selected subset of features
        feature_index, threshold = self._find_best_split(X_bootstrap, y_bootstrap, subset_features)

        if feature_index is not None:
            # Split the dataset
            X_left, X_right, y_left, y_right = self._split_dataset(X, y, feature_index, threshold)

            # Recursively build left and right subtrees
            left_subtree = self._build_tree(X_left, y_left, depth + 1)
            right_subtree = self._build_tree(X_right, y_right, depth + 1)

            return {'feature_index': feature_index,
                    'threshold': threshold,
                    'left': left_subtree,
                    'right': right_subtree}
        else:
            # If no split is found, return a leaf node
            return {'value': np.mean(y)}

    def _find_best_split(self, X, y, subset_features):
        """
        Find the best split for a subset of features.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - subset_features: Subset of features to consider.

        Returns:
        - Index of the best feature and the corresponding threshold.
        """
        m, n = X.shape
        best_feature_index = None
        best_threshold = None
        best_variance_reduction = 0

        initial_variance = self._calculate_variance(y)

        for feature_index in subset_features:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                # Split the dataset
                _, _, y_left, y_right = self._split_dataset(X, y, feature_index, threshold)

                # Calculate variance reduction
                left_weight = len(y_left) / m
                right_weight = len(y_right) / m
                variance_reduction = initial_variance - (left_weight * self._calculate_variance(y_left) + right_weight * self._calculate_variance(y_right))

                # Update the best split if variance reduction is greater
                if variance_reduction > best_variance_reduction:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_variance_reduction = variance_reduction

        return best_feature_index, best_threshold

    def _calculate_variance(self, y):
        """
        Calculate the variance of a set of target values.

        Parameters:
        - y: Target values (numpy array).

        Returns:
        - Variance of the target values.
        """
        return np.var(y)

    def _split_dataset(self, X, y, feature_index, threshold):
        """
        Split the dataset based on a feature and threshold.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - feature_index: Index of the feature to split on.
        - threshold: Threshold value for the split.

        Returns:
        - Left and right subsets of the dataset.
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def fit(self, X, y):
        """
        Fit the Random Forest Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Create a bootstrap sample for each tree
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)

            # Build a decision tree and add it to the forest
            tree = self._build_tree(X_bootstrap, y_bootstrap, depth=0)
            self.trees.append(tree)

    def _predict_single(self, tree, x):
        """
        Recursively predict a single data point using a decision tree.

        Parameters:
        - tree: Decision tree node.
        - x: Input features for prediction.

        Returns:
        - Predicted value.
        """
        if 'value' in tree:
            return tree['value']
        else:
            if x[tree['feature_index']] <= tree['threshold']:
                return self._predict_single(tree['left'], x)
            else:
                return self._predict_single(tree['right'], x)

    def predict(self, X):
        """
        Make predictions on new data using the Random Forest.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        predictions = np.array([self._predict_single(tree, x) for x in X for tree in self.trees])
        return np.mean(predictions.reshape(-1, len(self.trees)), axis=1)
```

- random_forest_regression_test.py file 

```py
import unittest
import numpy as np
from RandomForestRegressor import RandomForestRegression

class TestRandomForestRegressor(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = 2 * self.X_train[:, 0] + 3 * self.X_train[:, 1] + np.random.normal(0, 0.1, 100)

        self.X_test = np.random.rand(10, 2)

    def test_fit_predict(self):
        # Test if the model can be fitted and predictions are made
        rfr_model = RandomForestRegression(n_trees=5, max_depth=3, max_features=2)
        rfr_model.fit(self.X_train, self.y_train)

        # Ensure predictions are made without errors
        predictions = rfr_model.predict(self.X_test)

        # Add your specific assertions based on the expected behavior of your model
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (10,))

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
```
