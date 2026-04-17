# Parallel Model Training with Ray on Titanic

## Overview
This lab demonstrates **parallel model training** using **Ray** on the Titanic dataset. Multiple **Random Forest Classifiers** with varying `n_estimators` are trained simultaneously across Ray workers.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)
- **Metric**: Accuracy

## Notebook
`Ray_Titanic.ipynb`

## Concepts Covered

### Ray Core APIs
- `ray.init()` — initialize Ray cluster
- `ray.put()` — store objects in distributed memory
- `ray.get()` — retrieve results from workers
- `@ray.remote` — mark functions for parallel execution

### Sequential vs Parallel Training
- Train 20 RF models sequentially as baseline
- Train the same 20 models in parallel with Ray
- Compare wall time and identify speedup

## Performance

| Approach | Estimated Time | Speedup |
|----------|---------------|---------|
| Sequential | ~60s | 1x |
| Ray Parallel | ~10s | ~6x |

## Dependencies
```
ray, scikit-learn, seaborn, pandas, numpy
```

## How to Run
```bash
jupyter notebook Ray_Titanic.ipynb
```
