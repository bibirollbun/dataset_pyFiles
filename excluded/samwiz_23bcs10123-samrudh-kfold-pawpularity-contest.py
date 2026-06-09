import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data, num_splits, target_col="Pawpularity"):
    """
    Create stratified k-folds for regression problems
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe
    num_splits : int
        Number of folds
    target_col : str
        Name of target column
    """
    data = data.copy()  # Avoid modifying original data
    data["kfold"] = -1
    
    # Calculate number of bins
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # Alternative: use quantile-based binning for more balanced bins
    data.loc[:, "bins"] = pd.qcut(
        data[target_col], 
        q=num_bins, 
        labels=False, 
        duplicates='drop'
    )
    
    kf = model_selection.StratifiedKFold(
        n_splits=num_splits, 
        shuffle=True, 
        random_state=42
    )
    
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

