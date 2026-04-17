# Quantization and Pruning with Random Forest on Titanic

## Overview
This lab demonstrates **model compression** techniques for a **Random Forest Classifier** on the Titanic dataset. Neural network quantization and pruning concepts are adapted to tree-based models.

## Dataset
- **Titanic Dataset** — 891 passengers
- **Target**: `survived` (binary classification)

## Notebook
`Quantization_and_Pruning_Titanic.ipynb`

## Techniques Covered

### Post-Training Quantization
- Discretize continuous features into integer bins using `KBinsDiscretizer`
- Analog of float32 → int8 conversion in neural networks
- No retraining required

### Quantization-Aware Training (QAT)
- Train the model from scratch using quantized (binned) features
- Model adapts to reduced feature precision during training
- Typically yields better accuracy on quantized inputs

### Pruning
- Constrain tree growth via `max_depth`, `min_samples_split`, `min_samples_leaf`
- Reduces model complexity and improves compressibility
- Analog of weight sparsification in neural networks

### Combined Pruning + Quantization
- Apply both techniques for maximum compression
- Achieves smallest model size with acceptable accuracy

## Size Comparison (gzipped)

| Model | Relative Size |
|-------|--------------|
| Baseline (full RF) | 1x |
| Post-training quantized | ~same accuracy, smaller compressed |
| Pruned | ~3x smaller |
| Pruned + Quantized | smallest |

## Dependencies
```
scikit-learn, seaborn, pandas, numpy, matplotlib
```

## How to Run
```bash
jupyter notebook Quantization_and_Pruning_Titanic.ipynb
```
