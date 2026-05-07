# ============================================================
# 1. Library
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# Bagging
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor
)

# Random Forest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

# Base estimator
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

# ============================================================
# RANDOM FOREST - CLASSIFICATION
# 주요 parameter : n_estimators, max_features
# n_estimators : 학습할 tree model 개수
# max_features : random forests는 feature 개수 지정 필요 
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# ============================================================
# RANDOM FOREST - REGRESSION
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    max_features=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)
