import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data, num_splits=5):
    data = data.copy()
    data["kfold"] = -1

    kf = model_selection.StratifiedKFold(
        n_splits=num_splits, shuffle=True, random_state=42
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data["target"])):
        data.loc[valid_idx, "kfold"] = fold

    return data


df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

df_5 = create_folds(df, num_splits=5)
df_10 = create_folds(df, num_splits=10)
df_5.head()


df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)

