import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Constructor for the LogisticRegression class.

        Parameters:
        - learning_rate: The step size for gradient descent.
        - n_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters:
        - z: Linear combination of input features and weights.

        Returns:
        - Sigmoid of z.
        """
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features):
        """
        Initialize weights and bias.

        Parameters:
        - n_features: Number of input features.

        Returns:
        - Initialized weights and bias.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        """
        Fit the Logistic Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target labels (numpy array).
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for _ in range(self.n_iterations):
            # Linear combination of features and weights
            linear_combination = np.dot(X, self.weights) + self.bias

            # Predictions using the sigmoid function
            predictions = self._sigmoid(linear_combination)

            # Update weights and bias using gradient descent
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted labels (numpy array).
        """
        linear_combination = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_combination)

        # Convert probabilities to binary predictions (0 or 1)
        return np.round(predictions)
