# Distributed Training with Titanic and Random Forest

## Overview
This lab demonstrates **distributed training** using **Ray** with a **Random Forest Classifier** on the Titanic dataset. It adapts the multi-worker TensorFlow strategy concept to scikit-learn models using Ray's distributed computing framework.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)

## Files

| File | Description |
|------|-------------|
| `Distributed_Training_Titanic.ipynb` | Main notebook |
| `titanic.py` | Data loader and RF model builder module |
| `main.py` | Multi-worker distributed training script |

## Concepts Covered
1. **Single-worker training** — baseline RandomForestClassifier on one process
2. **Multi-worker configuration** — Ray cluster setup and resource detection
3. **Object store** — `ray.put()` / `ray.get()` for shared distributed memory
4. **Remote tasks** — `@ray.remote` decorator for parallel worker execution
5. **Multi-worker training** — distribute training across Ray workers

## Architecture

```
Ray Cluster
├── Worker 0 (chief) — trains RF with 50 trees
└── Worker 1         — trains RF with 50 trees
         ↓
   Combined results via ray.get()
```

## Dependencies
```
ray, scikit-learn, seaborn, pandas, numpy
```

## How to Run
```bash
# Single worker (notebook)
jupyter notebook Distributed_Training_Titanic.ipynb

# Multi-worker script
python main.py
```
