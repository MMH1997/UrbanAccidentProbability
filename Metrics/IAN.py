import numpy as np


def ia_n(y_true, probs):
    """
    Compute IA/N metric.

    IA/N = mean(predicted_probabilities | y=1) / mean(predicted_probabilities | y=0)

    Parameters
    ----------
    y_true : array-like
        True labels (0 or 1)
    probs : array-like
        Predicted probabilities for the positive class

    Returns
    -------
    float
        IA/N value
    """

    y_true = np.array(y_true)
    probs = np.array(probs)

    mean_accident = probs[y_true == 1].mean()
    mean_no_accident = probs[y_true == 0].mean()

    # Evitar división por cero
    if mean_no_accident == 0:
        return np.inf

    return mean_accident / mean_no_accident


def mean_probabilities(y_true, probs):
    """
    Return mean probabilities for each class.
    (Useful for debugging / analysis)
    """

    y_true = np.array(y_true)
    probs = np.array(probs)

    mean_accident = probs[y_true == 1].mean()
    mean_no_accident = probs[y_true == 0].mean()

    return mean_accident, mean_no_accident