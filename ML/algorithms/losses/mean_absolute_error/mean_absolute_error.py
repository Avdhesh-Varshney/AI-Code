# Algorithms/Losses/mean_squared_error.py
import numpy as np

def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error between true and predicted values.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Mean absolute error (float).
    """    
    return (np.absolute(y_true - y_pred)).mean()