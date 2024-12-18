import unittest
import numpy as np
from SVR import SupportVectorRegression

class TestSupportVectorRegression(unittest.TestCase):

    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

    def test_fit_predict(self):
        # Test the fit and predict methods
        svr_model = SupportVectorRegression(epsilon=0.1, C=1.0)
        svr_model.fit(self.X_train, self.y_train)
        
        # Create test data
        X_test = np.array([[1.5], [2.0]])
        
        # Make predictions
        predictions = svr_model.predict(X_test)
        
        # Assert that the predictions are NumPy arrays
        self.assertTrue(isinstance(predictions, np.ndarray))
        
        # Assert that the shape of predictions is as expected
        self.assertEqual(predictions.shape, (X_test.shape[0], 1))

if __name__ == '__main__':
    unittest.main()
