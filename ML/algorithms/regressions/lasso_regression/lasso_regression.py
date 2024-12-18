import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        """
        Constructor for the LassoRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - lambda_param: Regularization strength.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Lasso Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent
        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = y - predictions

            # Update weights and bias
            self.weights += self.learning_rate * (1/num_samples) * (np.dot(X.T, errors) - self.lambda_param * np.sign(self.weights))
            self.bias += self.learning_rate * (1/num_samples) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        return np.dot(X, self.weights) + self.bias
