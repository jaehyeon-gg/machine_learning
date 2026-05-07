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

# Base estimator - 모델 안에서 어떻게 학습할 건지 (지금은 decision tree 기반으로 학습 예정이라 decision tree 라이브러리 가져옴)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


# ============================================================
# BAGGING - CLASSIFICATION
# 주요 parameter : n_estimators : bagging 학습할 때 몇개의 tree를 만들건지. (bootstrap sample 몇개 만들건지)
# ============================================================

# X : feature table
# y : label

X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.2,random_state=42,stratify=y)

# Base model : (각 bootstrap sample이 학습할 tree model으로, 깊은 tree , variance 큰 tree 형성)
base_tree = DecisionTreeClassifier(max_depth=None,min_samples_leaf=1,random_state=42)

# Bagging model
model = BaggingClassifier(estimator=base_tree,n_estimators=100,max_samples=1.0,bootstrap=True, random_state=42,n_jobs=-1)

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
# BAGGING - REGRESSION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

base_tree = DecisionTreeRegressor(max_depth=None,random_state=42)

model = BaggingRegressor(estimator=base_tree,n_estimators=100,bootstrap=True,random_state=42,n_jobs=-1)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2:", r2)
