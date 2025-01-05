### **LightGBM**  
This module contains an implementation of LightGBM, a highly efficient gradient boosting algorithm optimized for speed and performance, especially for large datasets. It supports both classification and regression tasks.  

---

### **Parameters**  
- **num_leaves**: Maximum number of leaves in one tree.  
- **n_estimators**: Number of boosting rounds.  
- **learning_rate**: Step size at each iteration.  

---

### **Scratch Code**  

**`lightgbm_model.py`**  
```python
import lightgbm as lgb
import numpy as np

class LightGBMModel:
    def __init__(self, objective='regression', num_leaves=31, n_estimators=100, learning_rate=0.1):
        """
        LightGBM Model Constructor

        Parameters:
        - objective: 'regression' or 'binary' for classification
        - num_leaves: Max number of leaves per tree
        - n_estimators: Number of boosting rounds
        - learning_rate: Step size at each iteration
        """
        self.params = {
            'objective': objective,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate
        }
        self.n_estimators = n_estimators
        self.model = None
    
    def fit(self, X, y):
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data, num_boost_round=self.n_estimators)

    def predict(self, X):
        return self.model.predict(X)
```  

---

**`lightgbm_model_test.py`**  
```python
import unittest
import numpy as np
from lightgbm_model import LightGBMModel
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class TestLightGBMModel(unittest.TestCase):
    
    def test_regression(self):
        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        model = LightGBMModel(objective='regression', num_leaves=20, n_estimators=50)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Check if error is within acceptable range
        mse = mean_squared_error(y_test, y_pred)
        self.assertLess(mse, 4000)  # Example threshold

if __name__ == '__main__':
    unittest.main()
```  

---

### **Notes**  
- Make sure LightGBM is installed:  
  ```bash
  pip install lightgbm
  ``` 
