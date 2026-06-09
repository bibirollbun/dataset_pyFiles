# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os



from sklearn import datasets
from sklearn import model_selection


def create_folds(data, num_splits):
    data["kfold"] = -1
    num_bins = int(np.floor(1 + np.log2(len(data))))
    print('num_bins: ',num_bins)

    data.loc[:, "bins"] = pd.cut(data["Pawpularity"], bins=num_bins, labels=False)

    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    data = data.drop("bins", axis=1)

    return data


df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")

df_5 = create_folds(df, num_splits=5)
df_10 = create_folds(df, num_splits=10)


df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)


df_5.head()

