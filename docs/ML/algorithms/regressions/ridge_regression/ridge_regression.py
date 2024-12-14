import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Constructor for the Ridge Regression class.

        Parameters:
        - alpha: Regularization strength. Higher values specify stronger regularization.
        """
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        """
        Fit the Ridge Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Add a column of ones to the input features for the bias term
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Compute the closed-form solution for Ridge Regression
        identity_matrix = np.identity(X_bias.shape[1])
        self.weights = np.linalg.inv(X_bias.T @ X_bias + self.alpha * identity_matrix) @ X_bias.T @ y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        # Add a column of ones to the input features for the bias term
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # Make predictions using the learned weights
        predictions = X_bias @ self.weights

        return predictions
