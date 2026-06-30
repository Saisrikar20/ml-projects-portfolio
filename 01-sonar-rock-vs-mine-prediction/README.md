# Sonar Rock vs Mine Prediction using Machine Learning

A binary classification machine learning project that predicts whether an object detected by sonar signals is a **Rock (R)** or a **Mine (M)** using a Support Vector Machine (SVM).

This project demonstrates a complete machine learning workflow including data preprocessing, feature scaling, hyperparameter tuning, cross-validation, and model evaluation using Scikit-learn.

---

## Dataset

- **Dataset:** UCI Sonar Dataset
- **Samples:** 208
- **Features:** 60 numerical sonar frequency energy values
- **Classes:**
  - Rock (R)
  - Mine (M)

Each sample represents the energy reflected from sonar signals across 60 different frequency bands.

---

## Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-learn

---

## Machine Learning Pipeline

The project follows the standard supervised learning workflow.

```
Dataset
    │
    ▼
Train-Test Split
    │
    ▼
StandardScaler
    │
    ▼
Support Vector Machine (RBF Kernel)
    │
    ▼
GridSearchCV (Hyperparameter Tuning)
    │
    ▼
Model Evaluation
```

---

## Model

Algorithm:

- Support Vector Machine (SVM)
- RBF Kernel

Preprocessing:

- StandardScaler

Hyperparameter Optimization:

- GridSearchCV
- 5-Fold Cross Validation

Parameter Grid

```python
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__gamma": ["scale", "auto", 0.001, 0.01]
}
```

---

## Best Hyperparameters

```
C = 10
gamma = 0.01
```

---

## Results

| Metric | Score |
|---------|------:|
| Cross Validation Accuracy | **83.07%** |
| Training Accuracy | **100.00%** |
| Testing Accuracy | **90.48%** |

---

## Project Structure

```
Sonar-Rock-vs-Mine/
│
├── sonar.csv
├── sonar_prediction.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/sonar-rock-vs-mine.git
```

Move into the project directory

```bash
cd sonar-rock-vs-mine
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python sonar_prediction.py
```

---

## Example Output

```
Best params:      {'model__C': 10, 'model__gamma': 0.01}
Best CV accuracy: 0.8307

Train accuracy:   1.0000
Test accuracy:    0.9048
```

---

## Machine Learning Concepts Used

- Supervised Learning
- Binary Classification
- Train-Test Split
- Feature Scaling
- StandardScaler
- Pipeline
- Support Vector Machine (SVM)
- RBF Kernel
- Hyperparameter Tuning
- GridSearchCV
- Cross Validation
- Model Evaluation
- Accuracy Score

---

## Key Learnings

During development, several common machine learning mistakes were identified and corrected:

- Prevented data leakage by fitting only on the training data.
- Used `Pipeline` to ensure preprocessing was applied correctly.
- Applied `StandardScaler` before SVM.
- Used `GridSearchCV` for systematic hyperparameter tuning.
- Evaluated the final model on an unseen test dataset.

---

## Future Improvements

- Feature Selection
- Principal Component Analysis (PCA)
- Compare additional classifiers
  - Logistic Regression
  - Random Forest
  - XGBoost
- ROC Curve and AUC Evaluation
- Confusion Matrix Visualization

---

## Author

**Sai Srikar**

B.Tech Artificial Intelligence & Data Science

Machine Learning | Data Science | Cybersecurity