import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data, num_splits):
    data["kfold"] = -1
    num_bins = int(np.floor(1 + np.log2(len(data))))
    print('num_bins: ',num_bins)

    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)

    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
#     print(data.head())
    data = data.drop("bins", axis=1)

    return data


df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

df_5 = create_folds(df, num_splits=5)
df_10 = create_folds(df, num_splits=10)
df_5.head()


df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)




