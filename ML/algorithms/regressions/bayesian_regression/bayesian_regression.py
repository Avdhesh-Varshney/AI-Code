import numpy as np

class BayesianRegression:
    def __init__(self, alpha=1, beta=1):
        """
        Constructor for the BayesianRegression class.

        Parameters:
        - alpha: Prior precision.
        - beta: Noise precision.
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def fit(self, X, y):
        """
        Fit the Bayesian Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Add a bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute posterior precision and mean
        self.w_precision = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        self.w_mean = self.beta * np.linalg.solve(self.w_precision, X.T @ y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        # Add a bias term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute predicted mean
        y_pred = X @ self.w_mean

        return y_pred
