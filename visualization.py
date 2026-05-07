# ============================================================
# Library
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ============================================================
# Correlation Heatmap
# ============================================================

plt.figure(figsize=(8, 6))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.show()

# ============================================================
# Scatter Plot
# ============================================================

plt.figure(figsize=(6, 5))

sns.scatterplot(
    data=df,
    x="feature1",
    y="feature2",
    hue="target"
)

plt.title("Feature Scatter Plot")
plt.show()

# ============================================================
# Pairplot
# ============================================================

sns.pairplot(
    df,
    hue="target",
    diag_kind="kde"
)

plt.show()

# ============================================================
# Histogram
# ============================================================

plt.figure(figsize=(7, 5))

sns.histplot(
    data=df,
    x="feature1",
    hue="target",
    kde=True,
    bins=30
)

plt.title("Feature Distribution")
plt.show()

# ============================================================
# KDE Plot
# ============================================================

plt.figure(figsize=(7, 5))

sns.kdeplot(
    data=df,
    x="feature1",
    hue="target",
    fill=True
)

plt.title("KDE Distribution")
plt.show()

# ============================================================
# Box Plot
# ============================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="target",
    y="feature1"
)

plt.title("Box Plot")
plt.show()

# ============================================================
# PCA Visualization
# ============================================================

# 1. feature 선택 (numeric feature 선택)
X = df.drop(columns=["target"])

# 2. scaling (variance가 큰 data 방향으로 projection을 해야하기 때문에)
X_scaled = StandardScaler().fit_transform(X)

# 3. PCA 적용. 몇 차원으로 줄일건지 n_components 정하기
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "target": df["target"]
})

plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="target"
)

plt.title("PCA Projection")
plt.show()

