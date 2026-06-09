import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Temukan folder kompetisi secara otomatis
folder = os.listdir("/kaggle/input")[0]
csv_path = f"/kaggle/input/{folder}/penguins.csv"

# Load Dataset
df = pd.read_csv(csv_path)

# Data Cleaning
features = ['culmen_length_mm', 'culmen_depth_mm',
            'flipper_length_mm', 'body_mass_g']

# Hapus NA
df_clean = df.dropna(subset=features).copy()

# Hapus outlier ekstrem dengan IQR
Q1 = df_clean[features].quantile(0.25)
Q3 = df_clean[features].quantile(0.75)
IQR = Q3 - Q1

mask = ~((df_clean[features] < (Q1 - 1.5 * IQR)) | 
         (df_clean[features] > (Q3 + 1.5 * IQR))).any(axis=1)

df_clean = df_clean[mask].copy()

# Data siap pakai
X = df_clean[features]

# Standardisasi

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method

inertias = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.grid()
plt.show()

#  K-Means (k=3)
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

print("Jumlah anggota cluster:")
print(df_clean['cluster'].value_counts())

# Silhouette Score
sil = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", sil)

# PCA Visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

markers = ['o','s','^','x']

plt.figure()
for i in range(3):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        marker=markers[i],
        s=50,
        label=f"Cluster {i}"
    )

plt.title("PCA Visualization after Outlier Removal")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()


summary = df_clean.groupby('cluster')[features].agg(['mean','std','min','max'])
display(summary)

