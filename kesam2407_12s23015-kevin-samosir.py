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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns

# Untuk menampilkan plot lebih rapi
sns.set(style="whitegrid")



df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")  # Ganti sesuai nama filenya
df.head()



print(df.info())
print(df.isnull().sum())



df = df.dropna()  # Bisa juga pakai imputasi median jika ingin lebih baik
print(df.isnull().sum())



features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

X = df[features]



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Within-Cluster Sum of Squares

plt.plot(range(1, 10), wcss, marker='o')
plt.xlabel("Jumlah Cluster")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()



kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
df.head()



pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="kmeans_cluster", palette="viridis")
plt.title("K-Means Clustering Result (PCA)")
plt.show()



db = DBSCAN(eps=0.7, min_samples=5).fit(X_scaled)
df["dbscan_cluster"] = db.labels_
df["dbscan_cluster"].value_counts()



plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="dbscan_cluster", palette="rainbow")
plt.title("DBSCAN Clustering Result (PCA)")
plt.show()



# Membuat file submission untuk Kaggle
submission = pd.DataFrame({
    "id": df.index,
    "cluster": df['kmeans_cluster']  # atau dbscan_cluster sesuai pilihan kamu
})

submission.to_csv("submission.csv", index=False)
submission.head()





