import unittest
import numpy as np
from BayesianRegression import BayesianRegression

class TestBayesianRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

        self.X_test = 2 * np.random.rand(20, 1)

    def test_fit_predict(self):
        blr = BayesianRegression()
        blr.fit(self.X_train, self.y_train)
        y_pred = blr.predict(self.X_test)

        self.assertTrue(y_pred.shape == (20, 1))

if __name__ == '__main__':
    unittest.main()
