import numpy as np
import unittest
from LogisticRegression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 2)
        self.y_train = (np.random.rand(100) > 0.5).astype(int)

        self.X_test = np.random.rand(20, 2)

    def test_fit_predict(self):
        model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape, (20,))
        self.assertTrue(np.all(predictions == 0) or np.all(predictions == 1))

if __name__ == '__main__':
    unittest.main()
