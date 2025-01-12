# Mean Squared Error 

```py
import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error between true and predicted values.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Mean squared error (float).
    """
    return np.mean((y_true - y_pred) ** 2)
```
