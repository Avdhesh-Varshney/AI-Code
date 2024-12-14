import numpy as np
import math as mt

def root_mean_squared_error(y_true,y_pred):
    """
    Calculate the root mean squared error between true and predicted values.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Root Mean squared error (float).
    """
    return mt.sqrt(np.mean((y_true - y_pred) ** 2))

