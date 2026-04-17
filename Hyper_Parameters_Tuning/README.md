# Hyperparameter Tuning for Random Forest on Titanic

## Overview
This lab demonstrates **automated hyperparameter tuning** for a **Random Forest Classifier** on the Titanic dataset using **GridSearchCV** — the scikit-learn equivalent of Keras Tuner's Hyperband algorithm.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)

## Notebook
`Hyperparameter_Tuning_Titanic.ipynb`

## Concepts Covered

### Baseline Model
- RandomForestClassifier with manually selected hyperparameters
- Cross-validation accuracy as baseline benchmark

### GridSearchCV (Automated Hypertuning)
- Defines a hyperparameter search space
- Systematically tests all combinations using 5-fold cross-validation
- Selects the optimal set automatically

## Hyperparameters Tuned

| Hyperparameter | Search Space | Description |
|----------------|-------------|-------------|
| `n_estimators` | [50, 100, 200] | Number of trees |
| `max_depth` | [None, 5, 10] | Maximum tree depth |
| `min_samples_split` | [2, 5, 10] | Min samples to split a node |
| `min_samples_leaf` | [1, 2, 4] | Min samples in leaf node |

## Workflow
1. Train baseline model with default hyperparameters
2. Define hyperparameter search space
3. Run GridSearchCV to find optimal parameters
4. Rebuild and evaluate the hypertuned model
5. Compare baseline vs hypertuned performance

## Dependencies
```
scikit-learn, seaborn, pandas, numpy, matplotlib
```

## How to Run
```bash
jupyter notebook Hyperparameter_Tuning_Titanic.ipynb
```
