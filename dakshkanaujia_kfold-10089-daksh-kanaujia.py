import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(df, n_folds):
    
    # Initialize fold column
    df["fold"] = -1
    
    # Calculate number of bins using Sturges' formula
    bins = int(np.floor(1 + np.log2(len(df))))
    print(f"Number of bins: {bins}")
    
    # Bin the target variable to create stratified groups
    df["bins"] = pd.cut(df["Pawpularity"], bins=bins, labels=False)
    
    # Initialize stratified K-Fold
    skf = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Assign folds
    for fold_idx, (_, val_idx) in enumerate(skf.split(X=df, y=df["bins"])):
        df.loc[val_idx, "fold"] = fold_idx
    
    # Drop temporary bins column
    df.drop(columns=["bins"], inplace=True)
    
    return df


df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")

df_5 = create_folds(df, n_folds=5)
df_10 = create_folds(df, n_folds=10)


df_5.to_csv("train_5folds.csv", index=False)
df_10.to_csv("train_10folds.csv", index=False)


df_5.head()




