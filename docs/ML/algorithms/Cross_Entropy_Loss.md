# Cross Entropy Loss 

```py
import numpy as np

def binary_cross_entropy_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """
    Calculate the binary cross entropy loss between true and predicted values.
    It measures the difference between the predicted probability distribution and the actual binary label distribution.
    The formula for binary cross-entropy loss is as follows:

    L(y, ŷ) = -[y * log(ŷ) + (1 — y) * log(1 — ŷ)]

    where y is the true binary label (0 or 1), ŷ is the predicted probability (ranging from 0 to 1), and log is the natural logarithm.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Binary cross entropy loss (float).
    """
    if (y_true is not None) and (y_pred is not None):
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_true.shape == y_pred.shape, f"Shape of y_true ({y_true.shape}) does not match y_pred ({y_pred.shape})"
        # calculate the binary cross-entropy loss
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        return loss
    else:
        return None

def weighted_binary_cross_entropy_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list, w_pos: float, w_neg: float) -> float:
    """
    Calculates the weighted binary cross entropy loss between true and predicted values.
    Weighted Binary Cross-Entropy loss is a variation of the binary cross-entropy loss that allows for assigning different weights to positive and negative examples. This can be useful when dealing with imbalanced datasets, where one class is significantly underrepresented compared to the other.
    The formula for weighted binary cross-entropy loss is as follows:
    
    L(y, ŷ) = -[w_pos * y * log(ŷ) + w_neg * (1 — y) * log(1 — ŷ)]
    
    where y is the true binary label (0 or 1), ŷ is the predicted probability (ranging from 0 to 1), log is the natural logarithm, and w_pos and w_neg are the positive and negative weights, respectively.

    Parameters:
    - y_true: True target values (numpy array).
    - y_pred: Predicted values (numpy array).

    Returns:
    - Weighted binary cross entropy loss (float).
    """
    if (y_true is not None) and (y_pred is not None):
        assert w_pos != 0.0, f"Weight w_pos = {w_pos}"
        assert w_neg != 0.0, f"Weight w_neg = {w_neg}"
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_true.shape == y_pred.shape, f"Shape of y_true ({y_true.shape}) does not match y_pred ({y_pred.shape})"
        # calculate the binary cross-entropy loss
        loss = -(w_pos * y_true * np.log(y_pred) + w_neg * (1 - y_true) * np.log(1 - y_pred)).mean()
        return loss
    else:
        return None


def categorical_cross_entropy_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """
    Calculate the categorical cross entropy loss between true and predicted values.
    It measures the difference between the predicted probability distribution and the actual one-hot encoded label distribution.
    The formula for categorical cross-entropy loss is as follows:
    
    L(y, ŷ) = -1/N * Σ[Σ{y * log(ŷ)}]
    
    where y is the true one-hot encoded label vector, ŷ is the predicted probability distribution, and log is the natural logarithm.

    Parameters:
    - y_true: True target values (numpy array) (one-hot encoded).
    - y_pred: Predicted values (numpy array) (probabilities).

    Returns:
    - Categorical cross entropy loss (float).
    """
    if (y_true is not None) and (y_pred is not None):
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_pred.ndim == 2, f"Shape of y_pred should be (N, C), got {y_pred.shape}"
        assert y_true.shape == y_pred.shape, f"Shape of y_true ({y_true.shape}) does not match y_pred ({y_pred.shape})"
        
        # Ensure numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # calculate the categorical cross-entropy loss
        loss = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred)))
        return loss.mean()
    else:
        return None

def sparse_categorical_cross_entropy_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """
    Calculate the sparse categorical cross entropy loss between true and predicted values.
    It measures the difference between the predicted probability distribution and the actual class indices.
    The formula for sparse categorical cross-entropy loss is as follows:
    
    L(y, ŷ) = -Σ[log(ŷ[range(N), y])]
    
    where y is the true class indices, ŷ is the predicted probability distribution, and log is the natural logarithm.

    Parameters:
    - y_true: True target values (numpy array) (class indices).
    - y_pred: Predicted values (numpy array) (probabilities).

    Returns:
    - Sparse categorical cross entropy loss (float).
    """
    if (y_true is not None) and (y_pred is not None):
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_true.shape[0] == y_pred.shape[0], f"Batch size of y_true ({y_true.shape[0]}) does not match y_pred ({y_pred.shape[0]})"
        
        # convert true labels to one-hot encoding
        y_true_onehot = np.zeros_like(y_pred)
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

        # Ensure numerical stability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # calculate loss
        loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=-1))
        return loss
    else:
        return None


if __name__ == "__main__":
    # define true labels and predicted probabilities
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.3])

    print("\nTesting Binary Cross Entropy Loss")
    print("Y_True: ", y_true)
    print("Y_Pred:", y_pred)
    print("Binary Cross Entropy Loss: ", binary_cross_entropy_loss(y_true, y_pred))

    positive_weight = 0.7
    negative_weight = 0.3

    print("\nTesting Weighted Binary Cross Entropy Loss")
    print("Y_True: ", y_true)
    print("Y_Pred:", y_pred)
    print("Weighted Binary Cross Entropy Loss: ", weighted_binary_cross_entropy_loss(y_true, y_pred, positive_weight, negative_weight))

    y_true = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    print("\nTesting Categorical Cross Entropy Loss")
    print("Y_True: ", y_true)
    print("Y_Pred:", y_pred)
    print("Categorical Cross Entropy Loss: ", categorical_cross_entropy_loss(y_true, y_pred))

    y_true = np.array([1, 2, 0])
    y_pred = np.array([[0.1, 0.8, 0.1], [0.3, 0.2, 0.5], [0.4, 0.3, 0.3]])
    print("\nTesting Sparse Categorical Cross Entropy Loss")
    print("Y_True: ", y_true)
    print("Y_Pred:", y_pred)
    print("Sparse Categorical Cross Entropy Loss: ", sparse_categorical_cross_entropy_loss(y_true, y_pred))
```