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


Nama = "Kevin Gultom"
Nim = "12S23001"



# LOAD LIBRARIES & DATASET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")

df.head()




print(df.info())
print("\nDESKRIPSI DATA:")
print(df.describe())

print("\nJUMLAH MISSING VALUE:")
print(df.isnull().sum())



print(df.columns.tolist())




num_cols = ["culmen_length_mm", "culmen_depth_mm", 
            "flipper_length_mm", "body_mass_g"]

data = df[num_cols]

# Imputasi missing value dengan median
imputer = SimpleImputer(strategy="median")
data_imputed = imputer.fit_transform(data)

# Normalisasi (sangat penting untuk clustering)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

data_scaled[:5]



df_clean = df.dropna().reset_index(drop=True)

# Pilih fitur numerik (tanpa kolom sex)
features = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g"]

X = df_clean[features]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # >>> satu-satunya scaling yang dipakai



# Elbow & Silhouette
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
sil_scores = []
K = range(2, 8)

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow
plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia")
plt.show()

# Plot Silhouette
plt.figure(figsize=(6,4))
plt.plot(K, sil_scores, marker='o')
plt.title("Silhouette Score per k")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.show()


k_opt = 3  # sesuai hasil Elbow

kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_clean.loc[:, "cluster"] = cluster_labels
df_clean.head()


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_clean.loc[:, "pca1"] = X_pca[:, 0]
df_clean.loc[:, "pca2"] = X_pca[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_clean,
    x="pca1", y="pca2",
    hue="cluster",
    palette="viridis",
    s=60,
    edgecolor="black"
)

plt.xlim(-3, 4)
plt.ylim(df_clean["pca2"].min() - 0.5, df_clean["pca2"].max() + 0.5)

plt.title(f"PCA 2D – K-Means (k={k_opt})", fontsize=14)
plt.legend(title="Cluster")
plt.show()



cluster_summary = df_clean.groupby("cluster")[[
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]].mean()

print("Rata-rata fitur per cluster:\n")
cluster_summary


plt.figure(figsize=(12,6))
sns.boxplot(data=df_clean, x="cluster", y="body_mass_g", palette="viridis")
plt.title("Perbandingan Body Mass per Cluster")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df_clean, x="cluster", y="flipper_length_mm", palette="viridis")
plt.title("Perbandingan Flipper Length per Cluster")
plt.show()



print("\nKESIMPULAN CLUSTERING:\n")

for c in cluster_summary.index:
    row = cluster_summary.loc[c]
    
    print(f"Cluster {c}:")
    print(f"- Culmen Length rata-rata : {row['culmen_length_mm']:.2f}")
    print(f"- Culmen Depth rata-rata  : {row['culmen_depth_mm']:.2f}")
    print(f"- Flipper Length rata-rata: {row['flipper_length_mm']:.2f}")
    print(f"- Body Mass rata-rata     : {row['body_mass_g']:.2f}")
    print("")



submission = pd.DataFrame({
    "id": df_clean.index,
    "cluster": df_clean["cluster"]
})

# Simpan ke CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)

submission.head()


