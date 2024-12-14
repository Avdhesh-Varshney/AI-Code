import unittest
import numpy as np
from GradientBoostingRegressor import GradientBoostingRegression

class TestGradientBoostingRegressor(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = 2 * self.X_train[:, 0] + 3 * self.X_train[:, 1] + np.random.normal(0, 0.1, 100)

        self.X_test = np.random.rand(10, 2)

    def test_fit_predict(self):
        # Test if the model can be fitted and predictions are made
        gbr_model = GradientBoostingRegression(n_estimators=5, learning_rate=0.1, max_depth=3)
        gbr_model.fit(self.X_train, self.y_train)

        # Ensure predictions are made without errors
        predictions = gbr_model.predict(self.X_test)

        # Add your specific assertions based on the expected behavior of your model
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (10,))

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
