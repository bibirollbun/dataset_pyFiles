import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from typing import Counter


def apply_StratifiedGroupKFold(X, y, groups, n_splits, random_state=42):
    """Apply StratifiedGroupKFold cross-validation to a dataframe"""
    df_out = df.copy()

    # Apply StratifiedGroupKFold splitting
    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y, groups)):
        df_out.loc[val_index, "fold"] = fold_index

        # check
        train_tomo_ids, val_tomo_ids = groups[train_index], groups[val_index]
        assert len(set(train_tomo_ids) & set(val_tomo_ids)) == 0

    df_out = df_out.astype({"fold": 'int64'})
    return df_out


num_folds = 5

df = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")

df_out = apply_StratifiedGroupKFold(
    X=df,
    y=df["Number of motors"].values,
    groups=df["tomo_id"].values,
    n_splits=num_folds, 
    random_state=42,
)

df_out.to_csv(f"train_{num_folds}folds.csv", index=False)


import matplotlib.pyplot as plt
import seaborn as sns

cols = [
    #"Motor axis 0",
    #"Motor axis 1",
    #"Motor axis 2",
    "Array shape (axis 0)",
    "Array shape (axis 1)",
    "Array shape (axis 2)",
    "Voxel spacing",
    "Number of motors",
]
fig, axes = plt.subplots(ncols=1, nrows=len(cols), figsize=(12, 40))

for col, ax in zip(cols, axes):
    sns.countplot(x=col, data=df_out, hue="fold", ax=ax)

plt.show()

