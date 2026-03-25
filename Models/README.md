# Models

This directory contains the implementation of all machine learning models used in the project.

## Overview

Each model is implemented as a standalone Python module following a consistent structure. All models expose a common interface to:

* Train the model
* Predict probabilities
* Apply a classification threshold
* Compute basic statistics

This design allows easy comparison across models and seamless integration in the experimental notebooks.

## Implemented Models

* `CB.py` → Main model (CatBoost)
* `GB.py` → Gradient Boosting
* `RF.py` → Random Forest
* `LR.py` → Logistic Regression
* `SVM.py` → Support Vector Machine
* `KNN.py` → k-Nearest Neighbors
* `MLP.py` → Multi-Layer Perceptron

## Standard Usage

Each module provides a `run_<model>()` function that executes the full pipeline:

```python
from models.catboost_model import run_catboost

results = run_catboost(X_train, y_train, X_test, y_test)

print(results["mean_accident"])
print(results["mean_no_accident"])
```

## Output

Each model returns a dictionary with:

* `model` → trained model object
* `probs` → predicted probabilities
* `preds` → binary predictions (threshold applied)
* `mean_accident` → average predicted probability for positive class
* `mean_no_accident` → average predicted probability for negative class
* `training_time` → training time in seconds

## Notes

* All models operate on preprocessed data.
* Class imbalance handling (e.g., SMOTE) is expected to be applied before training.
* Evaluation metrics (including the IA/N index) are implemented in the `metrics/` directory.


