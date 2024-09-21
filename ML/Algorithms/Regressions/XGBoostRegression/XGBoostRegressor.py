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
