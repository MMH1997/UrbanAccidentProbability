from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss
)


# -----------------------------
# CLASSIFICATION METRICS
# -----------------------------
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def balanced_accuracy(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


# -----------------------------
# PROBABILITY-BASED METRICS
# -----------------------------
def roc_auc(y_true, probs):
    return roc_auc_score(y_true, probs)


def pr_auc(y_true, probs):
    return average_precision_score(y_true, probs)


def logloss(y_true, probs):
    return log_loss(y_true, probs)


def brier_score(y_true, probs):
    return brier_score_loss(y_true, probs)


# -----------------------------
# COMBINED EVALUATION
# -----------------------------
def evaluate_all(y_true, probs, threshold=0.1):
    """
    Compute all standard metrics.

    Parameters
    ----------
    y_true : array-like
    probs : array-like (predicted probabilities)
    threshold : float

    Returns
    -------
    dict
    """

    y_pred = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "roc_auc": roc_auc(y_true, probs),
        "pr_auc": pr_auc(y_true, probs),
        "logloss": logloss(y_true, probs),
        "brier_score": brier_score(y_true, probs),
    }