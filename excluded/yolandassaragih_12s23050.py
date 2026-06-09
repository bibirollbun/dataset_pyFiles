import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from sklearn.decomposition import PCA



df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")
print("Shape:", df.shape)
df.head(10)


df.info()
df.isna().sum()



# Hapus baris yang semua kolomnya NaN
df = df.dropna(how="all").copy()

num_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_cols = ["sex"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

X = df[num_cols + cat_cols]
X_prepared = preprocessor.fit_transform(X)

X_prepared.shape



inertias = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_prepared)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method untuk Menentukan k Optimal")
plt.show()



sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_prepared)
    sil_scores.append(silhouette_score(X_prepared, labels))

plt.figure()
plt.plot(list(K_range), sil_scores, marker="o")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score untuk Menentukan k Optimal")
plt.show()

best_k = list(K_range)[np.argmax(sil_scores)]
print("k terbaik berdasarkan silhouette =", best_k)



k_opt = best_k  # pakai hasil silhouette
kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init="auto")
labels = kmeans_final.fit_predict(X_prepared)

df["cluster"] = labels
df.head()



sse = kmeans_final.inertia_
sil = silhouette_score(X_prepared, labels)
dbi = davies_bouldin_score(X_prepared.toarray() if hasattr(X_prepared, "toarray") else X_prepared, labels)
chi = calinski_harabasz_score(X_prepared.toarray() if hasattr(X_prepared, "toarray") else X_prepared, labels)

print(f"SSE / Inertia           : {sse:.2f}")
print(f"Silhouette Score        : {sil:.4f} (lebih tinggi lebih baik)")
print(f"Davies-Bouldin Index    : {dbi:.4f} (lebih rendah lebih baik)")
print(f"Calinski-Harabasz Index : {chi:.2f} (lebih tinggi lebih baik)")



pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_prepared.toarray() if hasattr(X_prepared, "toarray") else X_prepared)

plt.figure(figsize=(7,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="viridis", s=50, alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Visualisasi Cluster K-Means (PCA 2D)")
plt.colorbar(label="Cluster")
plt.show()



cluster_summary = df.groupby("cluster")[num_cols].mean()
cluster_summary



# Jumlah anggota per cluster
df["cluster"].value_counts().sort_index()



# Boxplot sederhana per fitur (tanpa seaborn)
for col in num_cols:
    plt.figure()
    df.boxplot(column=col, by="cluster")
    plt.title(f"Distribusi {col} per Cluster")
    plt.suptitle("")
    plt.xlabel("Cluster")
    plt.ylabel(col)
    plt.show()



pca3 = PCA(n_components=3, random_state=42)
X_3d = pca3.fit_transform(X_prepared.toarray() if hasattr(X_prepared, "toarray") else X_prepared)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:,0], X_3d[:,1], X_3d[:,2], c=labels, cmap="viridis", s=40)
ax.set_title("Visualisasi Cluster (PCA 3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()


