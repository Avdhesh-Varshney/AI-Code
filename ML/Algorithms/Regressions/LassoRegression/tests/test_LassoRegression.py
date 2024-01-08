import unittest
import numpy as np
from LassoRegression import LassoRegression

class TestLassoRegression(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = 3 * self.X_train[:, 0] + 2 * self.X_train[:, 1] + np.random.randn(100)

        self.X_test = np.random.rand(10, 2)

    def test_fit_predict(self):
        # Test the fit and predict methods
        model = LassoRegression(learning_rate=0.01, lambda_param=0.1, n_iterations=1000)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        # Ensure predictions are of the correct shape
        self.assertEqual(predictions.shape, (10,))

if __name__ == '__main__':
    unittest.main()
