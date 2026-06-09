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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage



df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

# hapus missing value
df = df.dropna()

df.head()


df.info()
df.isna().sum()
df.describe()



df = df[df['flipper_length_mm'] > 0]   # hapus nilai -132
df = df.dropna()                       # hapus missing value



df.hist(figsize=(10,7))
plt.tight_layout()
plt.show()



import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # hanya menutup warning di block ini
    sns.pairplot(df, hue='sex')

plt.show()



features = ['culmen_length_mm','culmen_depth_mm','flipper_length_mm','body_mass_g']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance explained:", pca.explained_variance_ratio_)



plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.title("PCA of Penguin Features (Unlabeled)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



import warnings
warnings.filterwarnings("ignore")   # Opsional: hanya mematikan warning dalam cell ini

inertia = []
K = range(1, 10)

for k in K:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init='auto'      # solusi utama agar tidak ada warning sklearn
    )
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K, inertia, '-o')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()



kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
sil_kmeans



plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_kmeans, palette='viridis')
plt.title(f'K-Means Clusters (Silhouette={sil_kmeans:.3f})')
plt.show()



db = DBSCAN(eps=0.6, min_samples=5).fit(X_scaled)
labels_dbscan = db.labels_

sil_db = silhouette_score(X_scaled, labels_dbscan) if len(set(labels_dbscan))>1 else -1
sil_db

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_dbscan, palette='tab10')
plt.title("DBSCAN Clustering")
plt.show()



linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.show()



agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_scaled)

sil_agg = silhouette_score(X_scaled, labels_agg)
sil_agg



sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_agg, palette='viridis')
plt.title(f'Agglomerative Clustering (Silhouette={sil_agg:.3f})')
plt.show()



pd.DataFrame({
    'Method':['KMeans','DBSCAN','Agglomerative'],
    'Silhouette':[sil_kmeans, sil_db, sil_agg]
})





