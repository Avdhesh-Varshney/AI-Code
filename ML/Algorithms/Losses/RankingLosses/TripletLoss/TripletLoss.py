import tensorflow as tf
from typing import Tuple

def triplet_loss_func(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.3) -> tf.Tensor:
    """
    Computes the triplet loss for a batch of triplets.

    Args:
        y_true: True values of classification (unused in this implementation, typically required for compatibility with Keras).
        y_pred: Predicted values, expected to be a tensor of shape (batch_size, 3, embedding_dim) where 
                y_pred[:, 0] is the anchor, y_pred[:, 1] is the positive, and y_pred[:, 2] is the negative.
        alpha: Margin parameter for the triplet loss.

    Returns:
        loss: Computed triplet loss as a scalar tensor.
    """
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

    loss = tf.maximum(positive_dist - negative_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

# Example usage:
# model.compile(optimizer='adam', loss=triplet_loss_func)
