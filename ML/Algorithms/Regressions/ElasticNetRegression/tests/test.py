import unittest
import numpy as np
from sklearn.linear_model import ElasticNet
from ElasticNetRegression import ElasticNetRegression

class TestElasticNetRegression(unittest.TestCase):

    def test_elastic_net_regression(self):
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2  

        X_test = np.array([[2.5], [5.0], [7.5]])

        custom_model = ElasticNetRegression(alpha=1.0, l1_ratio=0.5)
        custom_model.fit(X_train, y_train)
        custom_predictions = custom_model.predict(X_test)
        sklearn_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)
        sklearn_model.fit(X_train, y_train)
        sklearn_predictions = sklearn_model.predict(X_test)

        np.testing.assert_allclose(custom_predictions, sklearn_predictions, rtol=1e-1)

        # Evaluate performance
        train_predictions_custom = custom_model.predict(X_train)
        train_predictions_sklearn = sklearn_model.predict(X_train)

        custom_mse = np.mean((y_train - train_predictions_custom) ** 2)
        sklearn_mse = np.mean((y_train - train_predictions_sklearn) ** 2)

        print(f"Custom Model MSE: {custom_mse}")
        print(f"Scikit-learn Model MSE: {sklearn_mse}")

if __name__ == '__main__':
    unittest.main()
