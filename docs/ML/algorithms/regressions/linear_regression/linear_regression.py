import numpy as np

# Linear regression implementation
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the LinearRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - n_iterations: The number of iterations for gradient descent.
        - n_iterations: n_epochs.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize weights and bias
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Compute predictions
            predictions = np.dot(X, self.weights) + self.bias

            # Calculate errors
            errors = predictions - y

            # Update weights and bias
            self.weights -= self.learning_rate * (1 / len(X)) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1 / len(X)) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        return np.dot(X, self.weights) + self.bias
