# Metrics

This directory contains the evaluation metrics used in the project.

## Overview

The metrics are divided into two groups:

### 1. IA/N (Main Metric)

The **IA/N index** is the main metric proposed in this work and is used for model comparison.

### 2. Standard Metrics

In addition to IA/N, the following standard metrics are included:

* Accuracy
* Balanced Accuracy
* ROC-AUC
* PR-AUC
* Log Loss
* Brier Score

These metrics include:

* Those used as **additional metrics** (Accuracy and Balanced Accuracy)
* Those used in **ablation and analysis experiments** (the remaining metrics)

## Usage

Standard metrics can be computed using:

```python
from metrics.metrics import evaluate_all

results = evaluate_all(y_true, probs)
```

IA/N is implemented separately.

