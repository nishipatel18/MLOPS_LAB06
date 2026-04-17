# MLOPS_LAB06 
A comprehensive MLOps lab replicating industry-standard ML workflows using the **Titanic dataset** and **Random Forest** models. Each folder covers a different model development technique.

---

## Repository Structure

```
MLOPS_LAB06/
├── Feature_Selection/          # Filter, Wrapper & Embedded feature selection
├── Distributed_Training/
│   ├── Lab1/                   # Multi-worker distributed training with Ray
│   └── Lab2/                   # Parallel model training with Ray
├── Hyper_Parameters_Tuning/    # Automated hyperparameter tuning (GridSearchCV)
├── Knowledge_Distillation/     # RF teacher → Decision Tree student distillation
├── Quantization_and_Pruning/   # Model compression: quantization + pruning
├── Fine_Tuning/                # Few-shot, Zero-shot, LoRA & Q-LoRA
└── Ray/                        # Parallel training with Ray
```

---

## Dataset

All labs use the **Titanic Dataset** (891 passengers, binary survival classification).

| Feature | Description |
|---------|-------------|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Gender (encoded: male=1, female=0) |
| `age` | Age in years |
| `sibsp` | Siblings/spouses aboard |
| `parch` | Parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation (encoded) |
| `survived` | **Target** — 0 = Not Survived, 1 = Survived |

---

## Labs

### Feature Selection
Compares six feature selection techniques to identify the most predictive features for survival:
- **Filter**: Pearson correlation, SelectKBest (ANOVA F-test)
- **Wrapper**: Recursive Feature Elimination (RFE)
- **Embedded**: Random Forest feature importance, L1 Regularization

### Distributed Training — Lab 1
Multi-worker distributed training using **Ray**. Simulates a two-worker cluster where each worker trains a Random Forest partition, demonstrating how distributed strategies scale model training.

### Distributed Training — Lab 2 / Ray
Trains 20 Random Forest models with increasing `n_estimators` (8–84) both sequentially and in parallel using Ray. Demonstrates a **~6x speedup** from parallelization.

### Hyperparameter Tuning
Automated hyperparameter search using **GridSearchCV** (scikit-learn equivalent of Keras Tuner). Tunes `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` with 5-fold cross-validation.

### Knowledge Distillation
Implements a custom **Distiller** class that transfers knowledge from a large Random Forest teacher (200 trees) to a shallow Decision Tree student (max_depth=5) using soft probability labels. The distilled student outperforms a student trained from scratch.

### Quantization and Pruning
Adapts neural network compression techniques to Random Forests:
- **Quantization** → Feature discretization with KBinsDiscretizer
- **Pruning** → Tree depth and leaf constraints
- **Combined** → Maximum compression with minimal accuracy loss

### Fine Tuning
Three notebooks adapting deep learning fine-tuning paradigms to Random Forests:
- **Few-Shot Learning** — Train with 5, 10, 20, 50, 100 examples and measure performance scaling
- **Zero-Shot Learning** — Train on 1st/2nd class, classify 3rd class without retraining
- **LoRA & Q-LoRA** — Parameter-efficient fine-tuning using warm_start adapter trees

---

## Dependencies

```bash
pip install scikit-learn pandas numpy seaborn matplotlib ray jupyter
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/nishipatel18/MLOPS_LAB06.git
cd MLOPS_LAB06

# Install dependencies
pip install scikit-learn pandas numpy seaborn matplotlib ray jupyter

# Open any lab
jupyter notebook Feature_Selection/Feature_Selection_Titanic.ipynb
```
