import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection


def create_folds(data: pd.DataFrame, num_splits: int = 5, target_col: str = "Pawpularity", random_state: int = 42) -> pd.DataFrame:
    """
    Create stratified K-folds for regression-like targets by binning continuous values.

    Args:
        data (pd.DataFrame): Input dataframe containing target_col.
        num_splits (int): Number of folds to create.
        target_col (str): Column name of the target variable.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with an added 'kfold' column indicating fold assignment.
    """
    
    # Copy to avoid modifying original data
    df = data.copy()
    df["kfold"] = -1

    # Determine number of bins using Sturges' rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    print(f"num_bins: {num_bins}")

    # Create bins for continuous target
    bins = pd.cut(df[target_col], bins=num_bins, labels=False)
    
    # Initialize StratifiedKFold
    skf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    
    # Assign fold numbers
    for fold, (_, val_idx) in enumerate(skf.split(df, bins)):
        df.loc[val_idx, "kfold"] = fold

    return df



df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")

df_5 = create_folds(df, num_splits=5)
df_10 = create_folds(df, num_splits=10)


df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)

