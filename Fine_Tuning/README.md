# Fine Tuning with Random Forest on Titanic

## Overview
This folder contains three notebooks that adapt **fine-tuning concepts** from deep learning to **Random Forest** models on the Titanic dataset.

## Notebooks

### 1. Few-Shot Learning (`Few_Shot_Learning_Titanic.ipynb`)
Trains a Random Forest with very limited labeled data (5, 10, 20, 50, 100 examples) and compares performance against the full-dataset baseline.

**Key concept**: How well does a model generalize when trained on minimal examples?

| Training Size | Expected Accuracy |
|--------------|-------------------|
| 5 examples | Low |
| 50 examples | Moderate |
| Full dataset (~712) | Best |

### 2. Zero-Shot Learning (`Zero_Shot_Learning_Titanic.ipynb`)
Trains on 1st+2nd class passengers and applies the model directly to 3rd class passengers without any retraining — demonstrating cross-distribution generalization.

**Key concept**: Can a model classify unseen distributions using only learned feature semantics?

### 3. LoRA & Q-LoRA (`LoRA_QLoRA_Titanic.ipynb`)
Adapts **Low-Rank Adaptation (LoRA)** to Random Forests using `warm_start` — freeze existing trees, train only a small "adapter" set of new trees.

**Key concept**: Fine-tune efficiently by updating only a fraction of model parameters.

| Approach | Trainable Trees | Parameter Reduction |
|----------|----------------|---------------------|
| Full retrain | 100 | 0% |
| LoRA | 10 | ~83% |
| Q-LoRA | 8 | ~87% + 4-bit quantization |

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)

## Dependencies
```
scikit-learn, seaborn, pandas, numpy, matplotlib
```

## How to Run
```bash
jupyter notebook Few_Shot_Learning_Titanic.ipynb
jupyter notebook Zero_Shot_Learning_Titanic.ipynb
jupyter notebook LoRA_QLoRA_Titanic.ipynb
```
