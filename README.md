# Metrics

This directory contains the evaluation metrics used throughout the project.

## Overview

The metrics are divided into two groups:

### 1. IA/N (Main Metric)

The **IA/N index** is the primary evaluation metric proposed in this work.
It measures how much higher the predicted probabilities are for accident events compared to non-accident events.

This metric is specifically designed for **rare event probability estimation**, where traditional classification metrics may fail to capture meaningful differences between models.

### 2. Standard Metrics

In addition to IA/N, a set of widely used metrics is implemented:

* Accuracy
* Balanced Accuracy
* ROC-AUC
* PR-AUC
* Log Loss
* Brier Score

These metrics serve two purposes:

* **Complementary evaluation**: to relate our results with standard approaches in the literature
* **Ablation and analysis experiments**: to assess model behavior under different experimental setups (e.g., by hour, by road)

## Usage

Standard metrics can be computed using:

```python
from metrics.metrics import evaluate_all

results = evaluate_all(y_true, probs)
```

IA/N is implemented separately and should be used as the main metric for model comparison.

