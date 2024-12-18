import numpy as np

# Polynomial regression implementation
class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the PolynomialRegression class.

        Parameters:
        - degree: Degree of the polynomial.
        - learning_rate: The step size for gradient descent.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _polynomial_features(self, X):
        """
        Create polynomial features up to the specified degree.

        Parameters:
        - X: Input features (numpy array).

        Returns:
        - Polynomial features (numpy array).
        """
        return np.column_stack([X ** i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fit the polynomial regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        X_poly = self._polynomial_features(X)
        self.weights = np.zeros((X_poly.shape[1], 1))
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X_poly, self.weights) + self.bias
            errors = predictions - y

            self.weights -= self.learning_rate * (1 / len(X_poly)) * np.dot(X_poly.T, errors)
            self.bias -= self.learning_rate * (1 / len(X_poly)) * np.sum(errors)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        X_poly = self._polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
