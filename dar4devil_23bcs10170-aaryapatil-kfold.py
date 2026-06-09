import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_stratified_folds(df, num_splits=5, target_col="Pawpularity", random_state=42):
    """
    Create stratified K-Folds for regression by binning the target variable.

    Args:
        df (pd.DataFrame): Input dataframe with the target column.
        num_splits (int): Number of folds.
        target_col (str): Name of the target column.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with a new column 'kfold' indicating fold assignment.
    """
    df = df.copy()
    df["kfold"] = -1

    # Use Sturges' rule to determine number of bins for stratification
    num_bins = int(np.floor(1 + np.log2(len(df))))
    print(f"Number of bins for stratification: {num_bins}")

    # Bin target values
    df["bins"] = pd.cut(df[target_col], bins=num_bins, labels=False)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["bins"])):
        df.loc[val_idx, "kfold"] = fold

    df = df.drop(columns=["bins"])
    return df


df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")

df_5fold = create_stratified_folds(df, num_splits=5)
df_10fold = create_stratified_folds(df, num_splits=10)


df_5fold.to_csv("train_5folds.csv", index=False)
df_10fold.to_csv("train_10folds.csv", index=False)


df_5fold.head()

