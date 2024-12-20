import numpy as np

class HuberRegression:
    def __init__(self, alpha=1.0, epsilon=1.35, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        learning_rate = 0.01

        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = y - y_pred

            # Compute Huber gradient
            mask = np.abs(error) <= self.epsilon
            gradient_w = (-2 / n_samples) * (X.T.dot(error * mask) + self.epsilon * np.sign(error) * (~mask)) + self.alpha * self.coef_
            gradient_b = (-2 / n_samples) * (np.sum(error * mask) + self.epsilon * np.sign(error) * (~mask))

            new_coef = self.coef_ - learning_rate * gradient_w
            new_intercept = self.intercept_ - learning_rate * gradient_b

            if np.all(np.abs(new_coef - self.coef_) < self.tol) and np.abs(new_intercept - self.intercept_) < self.tol:
                break

            self.coef_ = new_coef
            self.intercept_ = new_intercept

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_