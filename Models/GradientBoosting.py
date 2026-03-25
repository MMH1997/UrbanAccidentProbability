import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


def train_gradient_boosting(X_train, y_train, params=None):
    """
    Train a Gradient Boosting model.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    params : dict, optional

    Returns
    -------
    model : trained model
    training_time : float
    """

    if params is None:
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42
        }

    start_time = time.time()

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    return model, training_time


def predict_probabilities(model, X_test):
    """
    Predict probabilities for the positive class.
    """
    return model.predict_proba(X_test)[:, 1]


def apply_threshold(probs, threshold=0.1):
    """
    Convert probabilities to binary predictions using threshold.
    """
    return np.where(probs > threshold, 1, 0)


def compute_basic_stats(y_true, probs):
    """
    Compute mean probabilities for:
    - accident (y=1)
    - non-accident (y=0)
    """

    probs = np.array(probs)
    y_true = np.array(y_true)

    mean_accident = probs[y_true == 1].mean()
    mean_no_accident = probs[y_true == 0].mean()

    return mean_accident, mean_no_accident


def run_gradient_boosting(
    X_train,
    y_train,
    X_test,
    y_test,
    params=None,
    threshold=0.1
):
    """
    Full pipeline:
    - train
    - predict
    - threshold
    - compute stats
    """

    model, training_time = train_gradient_boosting(X_train, y_train, params)

    probs = predict_probabilities(model, X_test)
    preds = apply_threshold(probs, threshold)

    mean_acc, mean_no_acc = compute_basic_stats(y_test, probs)

    results = {
        "model": model,
        "probs": probs,
        "preds": preds,
        "mean_accident": mean_acc,
        "mean_no_accident": mean_no_acc,
        "training_time": training_time
    }

    return results