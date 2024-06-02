import numpy as np

class ElasticNetRegression:
    def __init__(self, learning_rate=0.01, l1_ratio=0.5, alpha=0.5, n_iterations=1000):
        """
        Constructor for the ElasticNetRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - l1_ratio: Ratio of L1 regularization in the penalty (0 for Ridge, 1 for Lasso).
        - alpha: Regularization strength.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the ElasticNet regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            # Update weights and bias with L1 and L2 regularization
            l1_regularization = self.l1_ratio * np.sign(self.weights)
            l2_regularization = (1 - self.l1_ratio) * self.weights
            self.weights -= self.learning_rate * (1 / n_samples) * (np.dot(X.T, errors) + self.alpha * (l1_regularization + l2_regularization))
            self.bias -= self.learning_rate * (1 / n_samples) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        return np.dot(X, self.weights) + self.bias