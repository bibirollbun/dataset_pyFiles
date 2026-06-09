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


import seaborn as sns
import matplotlib.pyplot as plt


train_data=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv') 
test_data=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submissions=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train_data.info()


train_data.isnull().sum()


train_data.describe()


sns.countplot(x='y',data=train_data)


categorical_features=[features for features in train_data.columns if train_data[features].dtype=='O']
numerical_features=[features for features in train_data.columns if train_data[features].dtype!='O']

## ---------- Distribution of Categorical Variables ----------
rows = int(np.ceil(len(categorical_features)/3))
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    sns.countplot(data=train_data, x=feature, ax=axes[idx], palette="Set2")
    axes[idx].set_title(f"Distribution of {feature} by Target (y)")
    axes[idx].tick_params(axis='x', rotation=45)

for idx in range(len(categorical_features), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()

# ---------- 2. Distribution of Numerical Variables ----------
rows = int(np.ceil(len(numerical_features)/3))
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(15, rows*4))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    sns.histplot(data=train_data, x=feature, kde=True, ax=axes[idx], element="step")
    axes[idx].set_title(f"Distribution of {feature} by Target (y)")

for idx in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[idx])
plt.tight_layout()
plt.show()

