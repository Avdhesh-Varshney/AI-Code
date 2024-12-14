import numpy as np
import unittest
from NeuralNetworkRegression import NeuralNetworkRegression

class TestNeuralNetworkRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 3)
        self.y_train = np.random.rand(100, 1)

        self.X_test = np.random.rand(10, 3)

    def test_fit_predict(self):
        # Initialize and fit the model
        model = NeuralNetworkRegression(input_size=3, hidden_size=4, output_size=1, learning_rate=0.01, n_iterations=1000)
        model.fit(self.X_train, self.y_train)

        # Ensure predictions have the correct shape
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape, (10, 1))

if __name__ == '__main__':
    unittest.main()
