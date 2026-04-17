# Feature Selection with Titanic Dataset

## Overview
This lab demonstrates various feature selection techniques applied to the Titanic survival prediction dataset using a **Random Forest Classifier**.

## Dataset
- **Titanic Dataset** — 891 passengers, 7 features after preprocessing
- **Target**: `survived` (0 = Not Survived, 1 = Survived)
- **Features**: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`

## Notebook
`Feature_Selection_Titanic.ipynb`

## Techniques Covered

| Method | Type | Description |
|--------|------|-------------|
| Pearson Correlation | Filter | Select features correlated with target (threshold = 0.2) |
| Inter-feature Correlation | Filter (Unsupervised) | Remove redundant correlated features |
| SelectKBest (F-test) | Filter | Select top 5 features using ANOVA F-statistic |
| RFE | Wrapper | Recursive Feature Elimination with RandomForest |
| Feature Importance | Embedded | Tree-based importance scores with SelectFromModel |
| L1 Regularization | Embedded | LinearSVC with L1 penalty for sparse selection |

## Evaluation Metrics
All feature subsets are evaluated using:
- Accuracy, ROC-AUC, Precision, Recall, F1 Score

## Dependencies
```
pandas, numpy, scikit-learn, seaborn, matplotlib
```

## How to Run
```bash
jupyter notebook Feature_Selection_Titanic.ipynb
```
