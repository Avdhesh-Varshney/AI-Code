# R2 Squared Error 

```py
import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared value between true and predicted values.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - R-squared value (float).
    """
    total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
    explained_variance = np.sum((y_pred - np.mean(y_true)) ** 2)
    r2 = 1 - (explained_variance / total_variance)
    return r2
```
