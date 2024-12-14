import numpy as np

class KNNRegression:
    def __init__(self, k=5):
        """
        Constructor for the KNNRegression class.

        Parameters:
        - k: Number of neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the KNN model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        predictions = []
        for x in X:
            # Calculate Euclidean distances between the input point and all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k-nearest neighbors
            indices = np.argsort(distances)[:self.k]

            # Average the target values of k-nearest neighbors
            predicted_value = np.mean(self.y_train[indices])
            predictions.append(predicted_value)

        return np.array(predictions)
