# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

#Mengabaikan runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



data = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
data.head()


data.info()
data.describe()


data.isnull().sum()


data = data.dropna()


print(data.columns.tolist())


#agar tiap fitur memiliki skala yang sama
from sklearn.preprocessing import StandardScaler

# Gunakan nama kolom sesuai dataset Kaggle
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Konversi hasil normalisasi menjadi DataFrame agar mudah divisualisasikan
scaled_df = pd.DataFrame(X_scaled, columns=features)

# Scatter plot dua fitur terpilih
plt.figure(figsize=(7,5))
plt.scatter(scaled_df["culmen_length_mm"], scaled_df["culmen_depth_mm"], alpha=0.7, color="teal")
plt.title("Sebaran Data Setelah Normalisasi (2 Fitur)")
plt.xlabel("Culmen Length (Standardized)")
plt.ylabel("Culmen Depth (Standardized)")
plt.grid(True)
plt.show()


# K=3 digunakan karena secara biologis ada 3 spesies penguin
kmeans_init = KMeans(n_clusters=3, random_state=42)
kmeans_init.fit(X_scaled)



inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan k Optimal')
plt.show()


for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k = {k}, silhouette score = {score:.3f}")


k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42)
labels_final = kmeans_final.fit_predict(X_scaled)
data['cluster'] = labels_final

print("\nDistribusi data per cluster:")
print(data['cluster'].value_counts())
print(f"\nInertia final (k={k_optimal}): {kmeans_final.inertia_:.2f}")


pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
data['pca1'], data['pca2'] = reduced[:,0], reduced[:,1]

plt.figure(figsize=(7,5))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='Set2', s=80)
plt.title("Visualisasi Cluster K-Means (PCA 2D)")
plt.show()


print("\n Evaluasi Klasterisasi:")
print(f"Inertia (Sum of Squared Error): {kmeans.inertia_:.2f}")

if "species" in data.columns:
    ari = adjusted_rand_score(data["species"], data["cluster"])
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")
    print("\nTabel Crosstab antara Species dan Cluster:")
    print(pd.crosstab(data["species"], data["cluster"]))
else:
    print("Kolom 'species' tidak ditemukan, ARI tidak dihitung.")


submission = data[features + ["cluster"]]
submission.to_csv("submission.csv", index=False)
print("\n File submission.csv berhasil dibuat.")

