# Sonar Rock vs Mine Prediction

Binary classification on the UCI Sonar dataset — distinguishing rocks from mines using sonar signal frequency responses. Built this as a hands-on project to get comfortable with the full sklearn ML workflow: data prep, model selection, cross-validation, and hyperparameter tuning.

---

## Results

| Metric | Score |
|--------|------:|
| CV Accuracy | **83.07%** |
| Train Accuracy | **100.00%** |
| Test Accuracy | **90.48%** |

Best hyperparameters found: `C=10`, `gamma=0.01`

---

## Dataset

- **Source:** UCI Sonar Dataset
- **Samples:** 208
- **Features:** 60 (sonar energy readings across frequency bands)
- **Target:** Rock (R) vs Mine (M)
- **Challenge:** High-dimensional, small-data problem — 60 features across 208 samples means models overfit easily

---

## Pipeline

```
Raw Data
    │
    ▼
Train-Test Split (80/20, stratified)
    │
    ▼
StandardScaler (inside Pipeline — no leakage)
    │
    ▼
SVM with RBF Kernel
    │
    ▼
GridSearchCV (5-fold CV, 20 param combinations)
    │
    ▼
Evaluation on held-out test set
```

---

## What I Learned Building This

Not a copy-paste project. Went through several model iterations and ran into real issues that forced me to understand what was happening under the hood:

- **Pipeline vs manual scaling** — fitting `StandardScaler` outside a Pipeline leaks validation data into the scaler during CV, inflating scores. Wrapping both in a `Pipeline` fixes this.
- **RandomForest on small data** — hit 100% training accuracy with default RF on 208 samples. Tree ensembles need a lot of data to generalize; SVM is the right tool here.
- **Data leakage via GridSearchCV** — called `grid.fit(x, y)` initially instead of `grid.fit(x_train, y_train)`. Got 100% test accuracy which was completely fake. Fixed by fitting only on training data.
- **CV score is the honest number** — test accuracy swings a lot on small datasets depending on the random split. 5-fold CV gives a more reliable estimate.

---

## Model Iterations

| Model | CV | Test | What Happened |
|-------|----|------|---------------|
| Logistic Regression (baseline) | — | 76.2% | Decent start, no tuning |
| LR + StandardScaler | — | 52.4% | Solver hit `max_iter` limit before converging |
| LR + Pipeline + CV | 66.4% | — | Right setup, model ceiling |
| RandomForest (default) | 66.4% | 76.2% | 100% train accuracy — overfitting |
| RandomForest (constrained) | 70.2% | 73.8% | Regularization helped CV, not enough |
| SVM default | ~64% | 78.6% | Right model, wrong hyperparameters |
| **SVM + GridSearchCV** | **83.1%** | **90.5%** | **Final result** |

---

## Project Structure

```
01-sonar-rock-vs-mine-prediction/
├── sonar_prediction.py
├── sonar.csv
├── requirements.txt
└── README.md
```

---

## Installation & Run

```bash
git clone https://github.com/Saisrikar20/ml-projects-portfolio.git
cd ml-projects-portfolio/01-sonar-rock-vs-mine-prediction
pip install -r requirements.txt
python sonar_prediction.py
```

---

## Output

```
Best params:      {'model__C': 10, 'model__gamma': 0.01}
Best CV accuracy: 0.8307

Train accuracy:   1.0000
Test accuracy:    0.9048
```

---

## Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sonar_data = pd.read_csv('sonar.csv', header=None)

x = sonar_data.drop(columns=60)
y = sonar_data[60]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=1
)

param_grid = {
    'model__C':     [0.01, 0.1, 1, 10, 100],
    'model__gamma': ['scale', 'auto', 0.001, 0.01]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  SVC(kernel='rbf'))
])

# fit on training data only — never grid.fit(x, y)
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x_train, y_train)

print(f"Best params:      {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
print(f"\nTrain accuracy:   {accuracy_score(y_train, best_model.predict(x_train)):.4f}")
print(f"Test accuracy:    {accuracy_score(y_test,  best_model.predict(x_test)):.4f}")
```

---

## Tech Stack

`Python 3` `scikit-learn` `pandas` `numpy`

---

## Author

**Sai Srikar B**
B.Tech AI & Data Science | Rajalakshmi Engineering College
[LinkedIn](https://linkedin.com/in/sai-srikar-b-54b1b8359) · [GitHub](https://github.com/Saisrikar20)
