import numpy as np


def acc(preds, ys):
    """
    Calculate accuracy, TPR, and TNR for binary predictions.

    Args:
        preds (list): List of predicted labels.
        ys (list): List of true labels.

    Returns:
        tuple: (accuracy, true_positive_rate, true_negative_rate)
    """
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    acc = np.mean(preds == ys)
    tpr = (
        np.sum((preds == 1) & (ys == 1)) / np.sum(ys == 1) if np.sum(ys == 1) > 0 else 0
    )
    tnr = (
        np.sum((preds == 0) & (ys == 0)) / np.sum(ys == 0) if np.sum(ys == 0) > 0 else 0
    )
    return acc, tpr, tnr