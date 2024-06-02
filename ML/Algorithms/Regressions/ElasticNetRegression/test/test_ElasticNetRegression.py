import unittest
import numpy as np
from ElasticNetRegression import ElasticNetRegression

class TestElasticNetRegression(unittest.TestCase):

    def setUp(self):
        # Set up some common data for testing
        np.random.seed(42)
        self.X_train = 2 * np.random.rand(100, 1)
        self.y_train = 4 + 3 * self.X_train + np.random.randn(100, 1)

        self.X_test = 2 * np.random.rand(20, 1)
        self.y_test = 4 + 3 * self.X_test + np.random.randn(20, 1)

    def test_fit_predict(self):
        # Test the fit and predict methods

        # Create an ElasticNetRegression model
        en_model = ElasticNetRegression()

        # Fit the model to the training data
        en_model.fit(self.X_train, self.y_train)

        # Make predictions on the test data
        predictions = en_model.predict(self.X_test)

        # Check that the predictions are of the correct shape
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_predict_with_unfitted_model(self):
        # Test predicting with an unfitted model

        # Create an ElasticNetRegression model (not fitted)
        en_model = ElasticNetRegression()

        # Attempt to make predictions without fitting the model
        with self.assertRaises(ValueError):
            _ = en_model.predict(self.X_test)

if __name__ == '__main__':
    unittest.main()