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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import PowerTransformer


sns.set_style('darkgrid')


data = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2022/data.csv')
data.head()


submission = pd.read_csv('/kaggle/input/tabular-playground-series-jul-2022/sample_submission.csv')


mask = np.triu(np.ones_like(data.corr(), dtype='bool'))
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), mask=mask, annot=True, fmt='.2f')


data = data.drop(columns='id')


data = data[
    ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28']]


cols = list(data.columns)


X_scaled = PowerTransformer().fit_transform(data)
X_scaled = pd.DataFrame(X_scaled, columns=cols)


pca = PCA(random_state=10, whiten=True)
X_pca = pca.fit_transform(X_scaled)
PCA_df = pd.DataFrame({'PCA_1': X_pca[:, 0], 'PCA_2': X_pca[:, 1]})
plt.figure(figsize=(14, 14))
sns.scatterplot(data=PCA_df, x='PCA_1', y='PCA_2', s=3)
plt.show()


gmm = BayesianGaussianMixture(n_components=7, n_init=5, covariance_type='full')
preds = gmm.fit_predict(X_scaled)


pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X_scaled)
df = pd.DataFrame({'x': reduced_data[:, 0], 'y': reduced_data[:, 1], 'clusters': preds})
plt.figure(figsize=(20, 10))
sns.scatterplot(x=df['x'], y=df['y'], hue=df['clusters'])
plt.show()


submission['Predicted'] = preds
submission.to_csv('submission.csv', index=False)




