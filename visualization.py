# ============================================================
# Library
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
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
# Violin Plot
# ============================================================

plt.figure(figsize=(7, 5))

sns.violinplot(
    data=df,
    x="target",
    y="feature1"
)

plt.title("Violin Plot")
plt.show()

# ============================================================
# PCA Visualization
# ============================================================

X = df.drop(columns=["target"])

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

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

# ============================================================
# t-SNE Visualization
# ============================================================

X = df.drop(columns=["target"])

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "target": df["target"]
})

plt.figure(figsize=(7, 6))

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="target"
)

plt.title("t-SNE Projection")
plt.show()
