
# Experiments

This directory contains the experimental setups used to evaluate the models.

## Overview

Three main experimental configurations are implemented using CatBoost as a reference model. These experiments can be generalized to other models.

## Experimental Setups

### 1. Main Experiment

A single global model is trained using:

- All available predictor variables  
- The full dataset  

This setup represents the baseline approach.

---

### 2. By-Hour Experiment

A separate model is trained for each hour of the day (24 models in total).

- Data is filtered by hour  
- Time-related variables are excluded  

This setup evaluates the impact of temporal specialization.

---

### 3. By-Road Experiment

A separate model is trained for each road (15 models in total).

- Data is filtered by road  
- The `Roads` variable is excluded  

This setup evaluates the impact of spatial specialization.

---

## Metrics

The following metrics are computed:

- IA/N (main metric)  
- Accuracy  
- Balanced Accuracy  

## Notes

- CatBoost is used as a reference model for all experiments.  
- The same experimental structure can be applied to other models.  
- Class imbalance is handled using SMOTE before training.
