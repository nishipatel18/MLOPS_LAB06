# Knowledge Distillation with Titanic Dataset

## Overview
This lab demonstrates **knowledge distillation** where a small `student` model learns from a more complex `teacher` model on the Titanic dataset. The teacher is a large **Random Forest** and the student is a shallow **Decision Tree**.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)
- **Split**: 70% train / 20% validation / 10% test

## Notebook
`Knowledge_Distillation_Titanic.ipynb`

## Concepts Covered

### Custom Distiller Class
Overrides `fit()`, `predict()`, and `evaluate()` with knowledge distillation logic:
- Teacher makes **soft probability predictions** (predict_proba)
- Soft labels are blended with hard labels: `α × hard + (1-α) × soft`
- Student learns from the blended signal

### Models

| Model | Architecture | Parameters |
|-------|-------------|------------|
| Teacher | RandomForestClassifier (200 trees) | Large, high accuracy |
| Student (distilled) | DecisionTreeClassifier (max_depth=5) | Small, guided by teacher |
| Student (scratch) | DecisionTreeClassifier (max_depth=5) | Small, trained alone |

### Key Parameters
- `alpha = 0.05` — weight for hard labels
- `temperature = 1.0` — controls softness of teacher predictions

## Results
- Teacher achieves highest accuracy
- Distilled student **outperforms** student trained from scratch
- Distilled student learns teacher's regularization patterns

## Dependencies
```
scikit-learn, seaborn, pandas, numpy, matplotlib
```

## How to Run
```bash
jupyter notebook Knowledge_Distillation_Titanic.ipynb
```
