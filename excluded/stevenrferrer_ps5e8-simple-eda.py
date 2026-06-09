from os import path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (KFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

import statsmodels.api as sm
import statsmodels.formula.api as smf


import warnings

# Ignore FutureWarnings from seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


sns.set_theme()
sns.set_palette('pastel')


BASE_PATH='/kaggle/input/playground-series-s5e8'
RNG_SEED = 42


train = pd.read_csv(path.join(BASE_PATH, 'train.csv'), index_col='id')
train


train.describe()


train.dtypes


all_features = train.drop('y', axis='columns').columns.tolist()
cat_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'default', 'housing', 'loan']
num_features = [x for x in all_features if x not in cat_features]

len(all_features), len(cat_features), len(num_features)


train.loc[train.duplicated()].shape


train.isnull().sum()


for cat_feature in cat_features+['y']:
    print(train.loc[:, cat_feature].value_counts())
    print()


cat_features


_, axes = plt.subplots(1, 2, figsize=(12, 3))

sns.countplot(
    data=train,
    x='job', hue='y',
    ax=axes[0]
)

sns.countplot(
    data=train, 
    x='marital', hue='y',
    ax=axes[1]
)

# slanted labels
for i in range(0, 2):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right');


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(
    data=train,
    x='education', hue='y',
    ax=axes[0]
)
sns.countplot(
    data=train, 
    x='contact', hue='y',
    ax=axes[1]
)

# slanted labels
for i in range(0, 2):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right');


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(
    data=train,
    x='month', hue='y',
    ax=axes[0]
)
sns.countplot(
    data=train, 
    x='poutcome', hue='y',
    ax=axes[1]
);

# slanted labels
for i in range(0, 2):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right');


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(
    data=train,
    x='default', hue='y',
    ax=axes[0]
)
sns.countplot(
    data=train, 
    x='loan', hue='y',
    ax=axes[1]
)

# slanted labels
for i in range(0, 2):
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right');


_, ax = plt.subplots(figsize=(8, 4))

sns.countplot(
    data=train,
    x='housing', hue='y',
    ax=ax
);

# slanted labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right');


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.kdeplot(
    data=train, 
    x='age', hue='y', 
    alpha=.4,
    fill=True,
    common_norm=False,
    ax=axes[0]
)

sns.kdeplot(
    data=train, 
    x='balance', hue='y', 
    alpha=.4,
    fill=True,
    common_norm=False,
    ax=axes[1]
);


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(
    data=train, 
    x='day', hue='y', 
    bins=range(1, 31), 
    ax=axes[0]
)

# duration values goes up to 4.9K, we show only up-to 1000
sns.histplot(
    data=train, 
    x='duration', hue='y', 
    bins=range(1, 1000), 
    ax=axes[1]
);


_, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.kdeplot(
    data=train, 
    x='campaign', hue='y', 
    alpha=.4,
    fill=True,
    common_norm=False,
    ax=axes[0]
)

sns.kdeplot(
    data=train, 
    x='pdays', hue='y', 
    alpha=.4,
    fill=True,
    common_norm=False,
    ax=axes[1]
);


_, ax = plt.subplots(figsize=(8, 4))

sns.kdeplot(
    data=train, 
    x='previous', hue='y', 
    alpha=.4,
    fill=True,
    common_norm=False,
    ax=ax
);


_, ax = plt.subplots(figsize=(12, 6))

c = train.loc[:, num_features+['y']].corr()
sns.heatmap(c, annot=True, ax=ax);

