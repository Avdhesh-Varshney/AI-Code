import numpy as np

class SupportVectorRegression:

    def __init__(self, epsilon=0.1, C=1.0):
        """
        Constructor for the SupportVectorRegression class.

        Parameters:
        - epsilon: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function.
        - C: Regularization parameter. The strength of the regularization is inversely proportional to C.
        """
        self.epsilon = epsilon
        self.C = C
        self.weights = None
        self.bias = None

    def _linear_kernel(self, X1, X2):
        """
        Linear kernel function.

        Parameters:
        - X1, X2: Input data (numpy arrays).

        Returns:
        - Linear kernel result (numpy array).
        """
        return np.dot(X1, X2.T)

    def _compute_kernel_matrix(self, X):
        """
        Compute the kernel matrix for the linear kernel.

        Parameters:
        - X: Input data (numpy array).

        Returns:
        - Kernel matrix (numpy array).
        """
        m = X.shape[0]
        kernel_matrix = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                kernel_matrix[i, j] = self._linear_kernel(X[i, :], X[j, :])

        return kernel_matrix

    def fit(self, X, y):
        """
        Fit the Support Vector Regression model to the input data.

        Parameters:
        - X: Input features (numpy array).
        - y: Target values (numpy array).
        """
        m, n = X.shape

        # Create the kernel matrix
        kernel_matrix = self._compute_kernel_matrix(X)

        # Quadratic programming problem coefficients
        P = np.vstack([np.hstack([kernel_matrix, -kernel_matrix]),
                       np.hstack([-kernel_matrix, kernel_matrix])])
        q = np.vstack([self.epsilon * np.ones((m, 1)) - y, self.epsilon * np.ones((m, 1)) + y])

        # Constraints matrix
        G = np.vstack([np.eye(2 * m), -np.eye(2 * m)])
        h = np.vstack([self.C * np.ones((2 * m, 1)), np.zeros((2 * m, 1))])

        # Solve the quadratic programming problem
        solution = np.linalg.solve(P, q)

        # Extract weights and bias
        self.weights = solution[:n]
        self.bias = solution[n]

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features for prediction (numpy array).

        Returns:
        - Predicted values (numpy array).
        """
        predictions = np.dot(X, self.weights) + self.bias
        return predictions
