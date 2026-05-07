
# ------------------------------------------
  # Boosting - Regression 
# ------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def bootstrap_estimation(
    X,
    y,
    model,
    statistic_fn,
    B=1000, # 학습할 tree 개수 
    random_state=0
):
    """
    Generic bootstrap framework
    """

    np.random.seed(random_state)
    n = len(X)
    stats = []

    for _ in range(B):
        idx = np.random.choice(n, size=n, replace=True)

        X_boot = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        y_boot = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]

        model.fit(X_boot, y_boot)

        stats.append(statistic_fn(model))

    return np.array(stats)
# ------------------------------------------
  # AdaBoost - Classification
  # 
# ------------------------------------------


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier(max_depth=1)

abc = AdaBoostClassifier(
    estimator=base,
    n_estimators=50,
    learning_rate=1.0,
    random_state=0
)

abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
