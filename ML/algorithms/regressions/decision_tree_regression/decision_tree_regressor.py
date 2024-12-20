import numpy as np

class DecisionTreeRegression:

    def __init__(self, max_depth=None):
        """
        Constructor for the DecisionTreeRegression class.

        Parameters:
        - max_depth: Maximum depth of the decision tree.
        """
        self.max_depth = max_depth
        self.tree = None

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

    def _find_best_split(self, X, y):
        """
        Find the best split for the dataset.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).

        Returns:
        - Index of the best feature and the corresponding threshold.
        """
        m, n = X.shape
        best_feature_index = None
        best_threshold = None
        best_variance_reduction = 0

        initial_variance = self._calculate_variance(y)

        for feature_index in range(n):
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

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        - depth: Current depth of the tree.

        Returns:
        - Node of the decision tree.
        """
        # Check if max depth is reached or if all target values are the same
        if depth == self.max_depth or np.all(y == y[0]):
            return {'value': np.mean(y)}

        # Find the best split
        feature_index, threshold = self._find_best_split(X, y)

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

    def fit(self, X, y):
        """
        Fit the Decision Tree Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_single(self, node, x):
        """
        Recursively predict a single data point.

        Parameters:
        - node: Current node in the decision tree.
        - x: Input features for prediction.

        Returns:
        - Predicted value.
        """
        if 'value' in node:
            return node['value']
        else:
            if x[node['feature_index']] <= node['threshold']:
                return self._predict_single(node['left'], x)
            else:
                return self._predict_single(node['right'], x)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        return np.array([self._predict_single(self.tree, x) for x in X])
