import numpy as np
import unittest
from RidgeRegression import RidgeRegression  # Assuming your RidgeRegression class is in a separate file

class TestRidgeRegression(unittest.TestCase):
    def test_fit_predict(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        X_train = np.random.rand(100, 2)
        y_train = 3 * X_train[:, 0] + 5 * X_train[:, 1] + 2 + 0.1 * np.random.randn(100)
        X_test = np.random.rand(20, 2)

        # Create a Ridge Regression model
        ridge_model = RidgeRegression(alpha=0.1)

        # Fit the model to training data
        ridge_model.fit(X_train, y_train)

        # Make predictions on test data
        predictions = ridge_model.predict(X_test)

        # Ensure the predictions have the correct shape
        self.assertEqual(predictions.shape, (20,))

    def test_invalid_alpha(self):
        # Check if an exception is raised for an invalid alpha value
        with self.assertRaises(ValueError):
            RidgeRegression(alpha=-1)

    # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()
