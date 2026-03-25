# Traffic Accident Probability Estimation

Rare-event accident probability estimation using machine learning.

## Structure
- `datasets/` → raw and processed data (not fully included due to size)  
- `models/` → implementations of all models  
- `metrics/` → IA/N and standard metrics  
- `experiments/` → main, by-hour and by-road setups  

## Models
CatBoost (main), Logistic Regression, SVM, KNN, MLP, Random Forest, Gradient Boosting  

## Experiments
- **Main** → global model using all data  
- **By-hour** → one model per hour  
- **By-road** → one model per road  

## Metrics
- **IA/N** (main metric)  
- Accuracy, Balanced Accuracy  
- ROC-AUC, PR-AUC, Log Loss, Brier  

## Data
Dataset not included due to size.  
Sources: Madrid Open Data + AEMET.  
A small sample is provided for testing.

## Usage
Install dependencies and run the notebooks in `experiments/`.
