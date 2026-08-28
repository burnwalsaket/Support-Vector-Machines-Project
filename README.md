# Iris Species Classification with Support Vector Machines

Classifying iris flower species from petal and sepal measurements using a Support Vector Machine, with hyperparameter tuning via GridSearchCV.

## Overview

This project uses the classic Iris flower dataset to build a multi-class classifier that distinguishes between three iris species — *setosa*, *versicolor*, and *virginica* — based on four physical measurements. It walks through EDA, a baseline SVM, and a grid-searched SVM to compare performance.

## Dataset

- **Source:** Fisher's Iris dataset (loaded via `seaborn.load_dataset('iris')`)
- **Size:** 150 samples, 50 per species
- **Target:** `species` (setosa, versicolor, virginica)
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm)

## Exploratory Data Analysis

- Pairplot of all four features colored by species to visually inspect class separability
- KDE plot of sepal width vs. sepal length for the *setosa* class to examine its density distribution

*Setosa* is visibly the most linearly separable species from the other two based on petal dimensions, while *versicolor* and *virginica* show some overlap.

## Methodology

1. **Train/test split:** 67/33 split (`test_size=0.33`)
2. **Baseline model:** `SVC()` with default parameters (RBF kernel, C=1, gamma='scale')
3. **Hyperparameter tuning:** `GridSearchCV` over:
   - `C`: [0.1, 1, 10, 100]
   - `gamma`: [1, 0.1, 0.01, 0.001]
   - 5-fold cross-validation, 16 candidate combinations (80 total fits)
4. **Evaluation:** Confusion matrix and classification report for both the baseline and grid-searched models

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline SVC (default params) | 98% | 0.98 |
| Grid-searched SVC | 94% | 0.95 |

**Key finding:** The default SVM actually outperformed the grid-searched version on this test set. With only one genuinely ambiguous point between *versicolor* and *virginica* in a small, easily-separable dataset, extensive tuning offered no real benefit and slightly overfit to the cross-validation folds rather than the specific train/test split. This is a useful illustration that grid search optimizes for cross-validated performance, not necessarily performance on any one held-out split — and that on small, clean datasets, default hyperparameters can already be close to optimal.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn (`SVC`, `GridSearchCV`, `train_test_split`, `classification_report`, `confusion_matrix`)

## Repository Structure

```
├── Support_Vector_Machines_Project.ipynb   # Full analysis notebook
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Support_Vector_Machines_Project.ipynb
```

*Note: the dataset is loaded directly via `seaborn.load_dataset('iris')`, so no external file is required.*
