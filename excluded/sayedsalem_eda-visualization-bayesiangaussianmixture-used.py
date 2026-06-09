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

from sklearn.preprocessing import PowerTransformer

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from sklearn.metrics import silhouette_score



dataset = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2022/data.csv')
dataset


sample_sub = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2022/sample_submission.csv')
sample_sub


dataset.info()


dataset.isna().sum()


dataset.duplicated().sum()


X = dataset.drop('id', axis=1)


# Let's explore the correlation between the features

corr = X.corr()

sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)


# Let's plot the distribution of the features and explore the ranges

for col in X:
  sns.histplot(X[col], kde=True)
  plt.show()


# We can use PowerTransformer()

X_scaled = pd.DataFrame(PowerTransformer().fit_transform(X),columns=X.columns)
X_scaled


# Try different values for hyperparameter k

k_values = range(2, 15)
inertia_values = []
silhouette_scores = []


for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))


plt.figure(figsize=(10, 6))

plt.subplot(1,2,1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()


best_k = 9

best_kmeans = KMeans(n_clusters=best_k, random_state=42)
best_kmeans.fit(X_scaled)

y_pred = best_kmeans.predict(X_scaled)
y_pred


gm_model = GaussianMixture(n_components=best_k, random_state=42)

y_pred = gm_model.fit_predict(X_scaled)


plt.figure(figsize=(30,6))

for i in range(gm_model.means_.shape[0]):
  plt.scatter(np.arange(X_scaled.shape[1]), gm_model.means_[i])

plt.xticks(ticks=np.arange(X_scaled.shape[1]), labels=X_scaled.columns, rotation=90)
plt.grid(True)
plt.show()


useless_cols = ['f_00','f_01','f_02','f_03','f_04','f_05','f_06','f_14','f_15','f_16','f_17','f_18','f_19','f_20','f_21']

X_scaled.drop(useless_cols, axis=1)


bgm_model = BayesianGaussianMixture(n_components=best_k,n_init=10, random_state=42, init_params='random', tol=1e-4)

y_pred = bgm_model.fit_predict(X_scaled)


submission1 = pd.DataFrame({'Id': dataset['id'], 'Predicted': y_pred})
submission1.to_csv('submission.csv', index=False)

