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


import os
print(os.listdir('/kaggle/working'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Display settings
pd.options.display.max_columns = 20
pd.options.display.width = 120


path = "/kaggle/input/penguin-clustering-analysis/penguins.csv"
 
df = pd.read_csv(path)
print('Data shape:', df.shape)
df.head()


features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
df_clean = df[features].dropna().reset_index(drop=True)
print('Clean data shape:', df_clean.shape)
df_clean.describe()



scaler = StandardScaler()
X = scaler.fit_transform(df_clean[features])
X[:5]


inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot
plt.figure()
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for K-Means')
plt.grid(True)
plt.show()


sil_scores = []
K_range2 = range(2, 9)
for k in K_range2:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    sil_scores.append(sil)

plt.figure()
plt.plot(list(K_range2), sil_scores, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-Means')
plt.grid(True)
plt.show()

best_k = K_range2[np.argmax(sil_scores)]
print('Best k by silhouette:', best_k)


k_final = int(best_k)
kmeans_final = KMeans(n_clusters=k_final, init='random', random_state=42, n_init=10)
labels_final = kmeans_final.fit_predict(X)

df_result = df_clean.copy()
df_result['cluster'] = labels_final
df_result['cluster'] = df_result['cluster'].astype(int)
print(df_result['cluster'].value_counts())

# Show first rows
df_result.head()


plt.figure(figsize=(8,6))
for label in sorted(df_result['cluster'].unique()):
    subset = df_result[df_result['cluster'] == label]
    plt.scatter(subset['culmen_length_mm'], subset['culmen_depth_mm'], label=f'Cluster {label}', alpha=0.6)
# plot centroids (transform back to original space)
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, edgecolor='k')
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.title(f'K-Means clusters (k={k_final})')
plt.legend()
plt.grid(True)
plt.show()


out_path = 'penguins_kmeans_clusters.csv'
df_result.to_csv(out_path, index=False)
print('Saved to', out_path)







