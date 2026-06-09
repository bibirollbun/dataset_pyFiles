import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
silhouette_score,
silhouette_samples,
davies_bouldin_score,
calinski_harabasz_score
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")
print("Shape:", df.shape)
df.head(10)


df.info()
df.isnull().sum()


df.describe()


df2 = df.copy()
plt.figure(figsize=(14,8))

for i, col in enumerate(["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], 1):

    plt.subplot(2, 2, i)
    sns.histplot(df2[col], kde=True)
    plt.title(f"Histogram {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(14,8))

for i, col in enumerate(["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot {col}")

plt.tight_layout()
plt.show()


sns.pairplot(df.dropna(), corner=True)
plt.show()


df_clean = df.copy()

df_clean = df_clean[
(df_clean["flipper_length_mm"] > 0) &
(df_clean["flipper_length_mm"] < 300)
]

df_clean = df_clean.drop(columns=["sex"], errors="ignore")

df_clean.shape


df2 = df.drop(['sex'], axis=1) 
df_clean = df2.dropna()         
df_clean.head()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

X_scaled[:5]


# Inisialisasi list penampung nilai inertia dan silhouette
inertias = []
sil_scores = []

K_RANGE = range(2, 10)

# Loop untuk mencari nilai terbaik k
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))


plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(K_RANGE, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.grid()

plt.subplot(1,2,2)
plt.plot(K_RANGE, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.grid()

plt.show()

sil_scores


optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50)
labels = kmeans.fit_predict(X_scaled)
df_result = df_clean.copy()
df_result["cluster"] = labels
df_result.head()


print("Silhouette:", silhouette_score(X_scaled, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_scaled, labels))
print("Calinski-Harabasz:", calinski_harabasz_score(X_scaled, labels))
print("Inertia:", kmeans.inertia_)


sil_val = silhouette_samples(X_scaled, labels)

plt.figure(figsize=(10,6))
y_lower = 10

for i in range(optimal_k):
    cluster_vals = sil_val[labels == i]
    cluster_vals.sort()
    size = len(cluster_vals)
    y_upper = y_lower + size

    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, alpha=0.6)
    plt.text(-0.05, (y_lower + y_upper) / 2, str(i))

    y_lower = y_upper + 10

plt.axvline(x=silhouette_score(X_scaled, labels), color='red', linestyle='--')
plt.title("Silhouette Plot per Cluster")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster Label")
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="viridis", s=70)
plt.title("PCA 2D Cluster Visualization")
plt.show()


cluster_summary = df_result.groupby("cluster").mean().round(2)
cluster_summary

