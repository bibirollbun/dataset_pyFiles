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


Nama = "Simorangkir Jonathan"
Nim = "12S23030"


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns



# path default di Kaggle
path = "/kaggle/input/penguin-clustering-analysis/penguins.csv"

df = pd.read_csv(path)
df.head()



df.info()
df.isna().sum()


features = ["culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g"]

df_clean = df.copy()

# 4.1 Isi NA numerik dengan median
for col in features:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# 4.2 Isi NA pada sex dengan modus (nilai yang paling sering muncul)
df_clean["sex"].fillna(df_clean["sex"].mode()[0], inplace=True)



Q1 = df_clean[features].quantile(0.25)
Q3 = df_clean[features].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

for col in features:
    df_clean[col] = df_clean[col].clip(lower[col], upper[col])

df_clean[features].describe()



scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])



sse = []
K = range(2, 10)   # coba k = 2..9

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # inertia = SSE

plt.figure(figsize=(8,5))
plt.plot(K, sse, marker="o")
plt.xlabel("Jumlah klaster (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method – SSE vs k")
plt.grid(True)
plt.show()



k_opt = 3  # kalau dari grafik elbow kamu lihat tekukan di 3

kmeans_test = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
labels_test = kmeans_test.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, labels_test)
print("Silhouette score untuk k =", k_opt, ":", sil)



k_opt = 3  # pakai hasil Elbow

kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df_clean["cluster"] = cluster_labels
df_clean.head()



sse_final = kmeans_final.inertia_
sil_final = silhouette_score(X_scaled, cluster_labels)

print("SSE akhir (k =", k_opt, "):", sse_final)
print("Silhouette akhir:", sil_final)



pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df_clean["pca1"] = X_pca[:, 0]
df_clean["pca2"] = X_pca[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_clean,
    x="pca1", y="pca2",
    hue="cluster",
    palette="viridis"
)
plt.title(f"PCA 2D – K-Means (k={k_opt})")
plt.legend(title="Cluster")
plt.show()



submission = pd.DataFrame({
    "id": df_clean.index,         # atau nama kolom ID asli kalau ada
    "cluster": df_clean["cluster"]
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()











