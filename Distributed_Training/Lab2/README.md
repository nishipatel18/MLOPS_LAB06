# Parallel Model Training with Ray on Titanic

## Overview
This lab demonstrates **parallel model training** using **Ray** to speed up training multiple **Random Forest Classifiers** with varying hyperparameters on the Titanic dataset.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)
- **Metric**: Accuracy

## Notebook
`Ray_Titanic.ipynb`

## Concepts Covered

### Sequential Training
- Train 20 Random Forest models with increasing `n_estimators` (8, 12, 16, ..., 84)
- Measure total wall time as baseline

### Parallel Training with Ray
- `ray.init()` — start local Ray cluster
- `ray.put()` — place dataset in distributed object store
- `ray.get()` — retrieve completed results
- `@ray.remote` — mark function for distributed execution
- Achieve ~6x speedup over sequential training

## Performance Comparison

| Approach | Time | Speedup |
|----------|------|---------|
| Sequential | ~60s | 1x |
| Parallel (Ray) | ~10s | ~6x |

## Dependencies
```
ray, scikit-learn, seaborn, pandas, numpy
```

## How to Run
```bash
jupyter notebook Ray_Titanic.ipynb
```
