import numpy as np

def hinge_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list)-> float:
    """
    Calculates the hinge loss between true and predicted values.

    The formula for hinge loss is as follows:

    L(y, ŷ) = max(0, 1 - y * ŷ)

    """
    if (y_true is not None) and (y_pred is not None):
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_true.shape[0] == y_pred.shape[0], f"Batch size of y_true ({y_true.shape[0]}) does not match y_pred ({y_pred.shape[0]})"
    
    # replacing 0 values to -1
    y_pred = np.where(y_pred == 0, -1, 1)
    y_true = np.where(y_true == 0, -1, 1)

     # Calculate loss
    loss = np.maximum(0, 1 - y_true * y_pred).mean()
    return loss

if __name__ == "__main__":
    # define true labels and predicted probabilities
    actual = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
    predicted = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])

    print("\nTesting Hinge Loss")
    print("Y_True: ", actual)
    print("Y_Pred:", predicted)
    print("Hinge Loss: ", hinge_loss(actual, predicted))
