import numpy as np

def kl_divergence_loss(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """
    Calculate the Kullback-Leibler (KL) divergence between two probability distributions.
    KL divergence measures how one probability distribution diverges from another reference probability distribution.

    The formula for KL divergence is:
    D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    where P is the true probability distribution and Q is the predicted probability distribution.

    Parameters:
    - y_true: True probability distribution (numpy array or list).
    - y_pred: Predicted probability distribution (numpy array or list).

    Returns:
    - KL divergence loss (float).
    """
    if (y_true is not None) and (y_pred is not None):
        if type(y_true) == list:
            y_true = np.asarray(y_true)
        if type(y_pred) == list:
            y_pred = np.asarray(y_pred)
        assert y_true.shape == y_pred.shape, f"Shape of p_true ({y_true.shape}) does not match q_pred ({y_pred.shape})"

        # Ensure numerical stability by clipping the probabilities
        y_true = np.clip(y_true, 1e-15, 1)
        y_pred = np.clip(y_pred, 1e-15, 1)
        
        # Normalize the distributions
        y_true /= y_true.sum(axis=-1, keepdims=True)
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
        
        # Calculate KL divergence
        kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=-1)
        return kl_div.mean()
    else:
        return None

if __name__ == "__main__":
    y_true = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]]) # True probability distributions
    y_pred = np.array([[0.1, 0.6, 0.3], [0.2, 0.5, 0.3]]) # Predicted probability distributions
    
    print("\nTesting Kullback Leibler Divergence Loss")
    print("Y_True: ", y_true)
    print("Y_Pred:", y_pred)
    print("Kullback Leibler Divergence Loss: ", kl_divergence_loss(y_true, y_pred))
