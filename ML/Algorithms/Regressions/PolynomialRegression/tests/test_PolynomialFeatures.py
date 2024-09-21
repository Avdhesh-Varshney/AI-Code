import unittest
import numpy as np
from PolynomialRegression import PolynomialFeatures

class TestPolynomialRegression(unittest.TestCase):

    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

    def test_fit_predict(self):
        # Test the fit and predict methods
        poly_model = PolynomialFeatures(degree=2)
        poly_model.fit(self.X_train, self.y_train)
        
        # Create test data
        X_test = np.array([[1.5], [2.0]])
        
        # Make predictions
        predictions = poly_model.predict(X_test)
        
        # Assert that the predictions are NumPy arrays
        self.assertTrue(isinstance(predictions, np.ndarray))
        
        # Assert that the shape of predictions is as expected
        self.assertEqual(predictions.shape, (X_test.shape[0], 1))

if __name__ == '__main__':
    unittest.main()
