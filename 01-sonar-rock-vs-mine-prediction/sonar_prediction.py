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

# Fix — fit only on training data
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x_train, y_train)

print(f"Best params:      {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
print(f"\nTrain accuracy:   {accuracy_score(y_train, best_model.predict(x_train)):.4f}")
print(f"Test accuracy:    {accuracy_score(y_test,  best_model.predict(x_test)):.4f}")