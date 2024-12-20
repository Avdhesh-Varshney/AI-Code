import unittest
import numpy as np
from KNearestNeighborsRegression import KNNRegression

class TestKNNRegression(unittest.TestCase):

    def test_knn_regression(self):
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.rand(100, 1) * 10
        y_train = 2 * X_train.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

        X_test = np.array([[2.5], [5.0], [7.5]])

        # Initialize and fit the KNN Regression model
        knn_model = KNNRegression(k=3)
        knn_model.fit(X_train, y_train)

        # Test predictions
        predictions = knn_model.predict(X_test)
        expected_predictions = [2 * 2.5, 2 * 5.0, 2 * 7.5]  # Assuming a linear relationship

        # Check if predictions are close to the expected values
        np.testing.assert_allclose(predictions, expected_predictions, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
