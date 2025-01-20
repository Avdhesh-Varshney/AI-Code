# AdaBoost 

**Overview:**  
AdaBoost (Adaptive Boosting) is one of the most popular ensemble methods for boosting weak learners to create a strong learner. It works by combining multiple "weak" models, typically decision stumps, and focusing more on the errors from previous models. This iterative process improves accuracy and reduces bias.

---

### **Key Highlights:**  
- **Boosting Concept:** Builds an ensemble by sequentially focusing on harder-to-classify instances.  
- **Adaptive Weighting:** Misclassified instances get higher weights, and correctly classified instances get lower weights in subsequent rounds.  
- **Simple and Effective:** Often uses decision stumps (single-level decision trees) as base models.  
- **Versatility:** Applicable to both regression and classification problems.  

---

### **How AdaBoost Works (Scratch Implementation Guide):**  

#### **1. Core Concept (Error Weight Adjustment):**  
- Assigns equal weights to all data points initially.  
- In each iteration:
  - A weak model (e.g., a decision stump) is trained on the weighted dataset.  
  - Misclassified points are assigned higher weights for the next iteration.  
  - A final strong model is constructed by combining all weak models, weighted by their accuracy.  

*Visualization:*  
```  
Iteration 1: Train weak model -> Update weights  
Iteration 2: Train weak model -> Update weights  
...  
Final Model: Combine weak models with weighted contributions  
```

---

### **Algorithm Breakdown:**  
1. **Initialize Weights:** Assign equal weights to all instances.  
2. **Train a Weak Model:** Use weighted data to train a weak learner.  
3. **Calculate Model Error:** Compute the weighted error rate of the model.  
4. **Update Instance Weights:** Increase weights for misclassified points and decrease weights for correctly classified points.  
5. **Update Model Weight:** Calculate the model’s contribution based on its accuracy.  
6. **Repeat for a Set Number of Iterations or Until Convergence.**  

---

### **Parameters Explained:**  
- **n_estimators:** Number of weak learners (iterations).  
- **learning_rate:** Shrinks the contribution of each weak learner to avoid overfitting.  
- **base_estimator:** The weak learner used (e.g., `DecisionTreeRegressor` or `DecisionTreeClassifier`).  

---

### **Scratch Code Example (From the Ground Up):**  

**File:** `adaboost_model.py`  
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class AdaBoostRegressor:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train weak model
            model = DecisionTreeRegressor(max_depth=1)
            model.fit(X, y, sample_weight=sample_weights)
            predictions = model.predict(X)

            # Calculate weighted error
            error = np.sum(sample_weights * (y != predictions)) / np.sum(sample_weights)
            if error > 0.5:
                break

            # Calculate model weight
            model_weight = self.learning_rate * np.log((1 - error) / error)

            # Update sample weights
            sample_weights *= np.exp(model_weight * (y != predictions))
            sample_weights /= np.sum(sample_weights)

            self.models.append(model)
            self.model_weights.append(model_weight)

    def predict(self, X):
        # Combine predictions from all models
        final_prediction = sum(weight * model.predict(X) for model, weight in zip(self.models, self.model_weights))
        return np.sign(final_prediction)
```  

---

### **Testing the Model:**  

**File:** `adaboost_model_test.py`  
```python
import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from adaboost_model import AdaBoostRegressor

class TestAdaBoostRegressor(unittest.TestCase):

    def test_adaboost(self):
        # Generate synthetic dataset
        X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
        y = np.sign(y)  # Convert to classification-like regression

        # Train AdaBoost Regressor
        model = AdaBoostRegressor(n_estimators=10)
        model.fit(X, y)

        # Predict and Evaluate
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        self.assertTrue(mse < 0.5, "MSE is too high, AdaBoost not performing well")

if __name__ == '__main__':
    unittest.main()
```  

---

### **Additional Insights to Aid Understanding:**  
- **Feature Importance:**  
```python
for i, model in enumerate(model.models):
    print(f"Model {i} weight: {model_weights[i]}")
```  
- **Early Stopping Implementation:**  
Use validation metrics to stop training if performance does not improve over several iterations.  

---

### **Testing and Validation:**  
Use datasets from `sklearn` (e.g., `make_regression`) to validate the implementation. Compare AdaBoost with other boosting models like Gradient Boosting and LightGBM to analyze performance differences.
