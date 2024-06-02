import numpy as np

class ElasticNetRegression:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        learning_rate = 0.01  # This can be tuned

        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = y - y_pred

            gradient_w = (-2 / n_samples) * (X.T.dot(error)) + self.alpha * (self.l1_ratio * np.sign(self.coef_) + (1 - self.l1_ratio) * 2 * self.coef_)
            gradient_b = (-2 / n_samples) * np.sum(error)

            new_coef = self.coef_ - learning_rate * gradient_w
            new_intercept = self.intercept_ - learning_rate * gradient_b

            if np.all(np.abs(new_coef - self.coef_) < self.tol) and np.abs(new_intercept - self.intercept_) < self.tol:
                break

            self.coef_ = new_coef
            self.intercept_ = new_intercept

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
