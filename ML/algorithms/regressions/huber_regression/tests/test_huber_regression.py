import unittest
import numpy as np
from sklearn.linear_model import HuberRegressor
from HuberRegression import HuberRegression

class TestHuberRegression(unittest.TestCase):

    def test_huber_regression(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2

        X_test = np.array([[2.5], [5.0], [7.5]])

        huber_model = HuberRegression(alpha=1.0, epsilon=1.35)
        huber_model.fit(X_train, y_train)
        huber_predictions = huber_model.predict(X_test)

        sklearn_model = HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=1000, tol=1e-4)
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)

        np.testing.assert_allclose(huber_predictions, sklearn_predictions, rtol=1e-1)

        train_predictions_huber = huber_model.predict(X_train)
        train_predictions_sklearn = sklearn_model.predict(X_train)

        huber_mse = np.mean((y_train - train_predictions_huber) ** 2)
        sklearn_mse = np.mean((y_train - train_predictions_sklearn) ** 2)

        print(f"Huber Model MSE: {huber_mse}")
        print(f"Scikit-learn Model MSE: {sklearn_mse}")

if __name__ == '__main__':
    unittest.main()