import unittest
import numpy as np
from XGBoostRegressor import XGBoostRegressor

class TestXGBoostRegressor(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100)
        self.X_test = np.random.rand(20, 5)

    def test_fit_predict(self):
        # Test the fit and predict methods
        xgb_model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, gamma=0.1)
        xgb_model.fit(self.X_train, self.y_train)
        predictions = xgb_model.predict(self.X_test)

        # Ensure predictions have the correct shape
        self.assertEqual(predictions.shape, (20,))

    def test_invalid_parameters(self):
        # Test invalid parameter values
        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=-1, learning_rate=0.1, max_depth=3, gamma=0.1)

        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=50, learning_rate=-0.1, max_depth=3, gamma=0.1)

        with self.assertRaises(ValueError):
            XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=-3, gamma=0.1)

    def test_invalid_fit(self):
        # Test fitting with mismatched X_train and y_train shapes
        xgb_model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, gamma=0.1)
        with self.assertRaises(ValueError):
            xgb_model.fit(self.X_train, np.random.rand(50))

if __name__ == '__main__':
    unittest.main()
