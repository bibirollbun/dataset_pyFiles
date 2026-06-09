%load_ext cudf.pandas

import cudf
import cupy as cp
from cuml.model_selection import train_test_split
from cuml.linear_model import ElasticNet
from cuml.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Union
import warnings

# warnings kindly taken from: 
# https://www.kaggle.com/competitions/playground-series-s5e4/discussion/571034
msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


start = time.time()


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
target = "Listening_Time_minutes"


train


test[target] = np.nan # adding a pseudo target column so df schemas match
train["source"] = "train"
test["source"] = "test"
full_df = pd.concat([train, test]).reset_index(drop=True)


def calculate_groupwise_zscore(
    df: pd.DataFrame, 
    numeric_col: str, 
    group_cols: List[str],
    new_col_prefix: str
) -> pd.DataFrame:
    """
    Calculate the z-scores of 'numeric_col' within each group defined
    by 'group_cols' in the given DataFrame. Return the DataFrame
    with an additional column containing the z-scores.
    
    :param df: The DataFrame containing the data.
    :param numeric_col: Name of the numeric column for which we want to calculate z-scores.
    :param group_cols: List of categorical column names used to define the groups.
    :param new_col_prefix: Prefix for new column (i.e. to prevent duplicates)
    
    Returns the original DataFrame with an additional column '<numeric_col>_zscore'
    containing group-wise z-scores.
    """
    grouped = df.groupby(group_cols)
    
    # Calculate mean and std for the numeric column within each group
    mean_series = grouped[numeric_col].transform('mean')
    std_series = grouped[numeric_col].transform('std')
    
    # Compute the z-score and handle cases where std is 0 or NaN
    zscore_col = f"{new_col_prefix}_{numeric_col}_zscore"
    df[zscore_col] = (df[numeric_col] - mean_series) / std_series
    df[zscore_col] = df[zscore_col].fillna(0)
    
    return df


full_df = calculate_groupwise_zscore(
    full_df, 
    "Guest_Popularity_percentage", 
    ["Publication_Day", "Publication_Time"],
    "grby_pubday_pubtime_"
)


full_df = calculate_groupwise_zscore(
    full_df, 
    "Guest_Popularity_percentage", 
    ["Podcast_Name"],
    "grby_pdcastname_"
)


full_df = calculate_groupwise_zscore(
    full_df, 
    "Episode_Length_minutes", 
    ["Publication_Day", "Publication_Time"],
    "grby_pubday_pubtime_"
)


full_df = calculate_groupwise_zscore(
    full_df, 
    "Episode_Length_minutes", 
    ["Podcast_Name"],
    "grby_pdcastname_"
)


train = full_df.loc[(full_df["source"] == "train")].copy()
test = full_df.loc[(full_df["source"] == "test")].copy()
test = test.drop(target, axis=1)


class TargetEncoder:
    """
    Perform mean target encoding for one or more categorical columns, including
    a smoothing factor to reduce overfitting.
    
    The formula for the smoothed mean target encoding for each category is:
        encoding = (count * category_mean + alpha * global_mean) / (count + alpha)
    """
    def __init__(self, cat_cols: List[str], alpha: float = 10.0):
        """
        :param cat_cols: List of column names to encode.
        :param alpha: The smoothing factor. Higher values place more weight on the overall mean,
            while lower values place more weight on the category mean.
        """
        self.cat_cols = cat_cols
        self.alpha = alpha
        self.global_mean_ = None
        self.encodings_ = {}
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> "TargetEncoder":
        """
        Fit the target encoder to the training data. Learns the smoothed
        mean target for each category in the specified columns.

        :param X: The input DataFrame containing the data.
        :param y: The targets
        :param target_col: target column name.

        Returns fitted encoder with learned mappings in `self.encodings_`.
        """
        # Convert y to a Series if it's a DataFrame with a single column
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0]
        
        # Compute the global mean (float)
        self.global_mean_ = y.mean()
        
        # Combine X and y so we can group by columns in X while aggregating y
        df_temp = X.copy()
        df_temp["_target_"] = y.values
        
        # For each categorical column, compute the smoothed means
        for col in self.cat_cols:
            # Aggregate the target mean and count by the categorical column
            agg_df = df_temp.groupby(col)["_target_"].agg(["mean", "count"])
            
            # Compute smoothed mean
            agg_df["smooth"] = (
                (agg_df["count"] * agg_df["mean"] + self.alpha * self.global_mean_)
                / (agg_df["count"] + self.alpha)
            )
            
            # Store the mapping of category -> smooth mean
            self.encodings_[col] = agg_df["smooth"]
        
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the given DataFrame by mapping each category in the
        specified columns to the learned smoothed mean.

        :param X: The data to transform.
        
        Returns a copy of the original DataFrame with additional columns
            <column>_te for each encoded categorical feature.
        """
        if not self.fitted_:
            raise ValueError("This TargetEncoder instance is not fitted yet. "
                             "Call 'fit' before using 'transform'.")
        
        X_enc = X.copy()
        
        # Map the learned smooth means; if a category wasn't seen in fit,
        # fill with global_mean_
        for col in self.cat_cols:
            te_col = f"{col}_te"
            smooth_series = self.encodings_[col]
            X_enc[te_col] = X_enc[col].map(smooth_series).fillna(self.global_mean_)
        
        return X_enc
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        :param X: The training features.
        :param y: The target values.
        
        Returns transformed DataFrame with target-encoded columns.
        """
        return self.fit(X, y).transform(X)



encoder = TargetEncoder(
    cat_cols=[
        "Podcast_Name", 
        "Episode_Title", 
        "Genre", 
        "Publication_Day", 
        "Number_of_Ads", 
        "Episode_Sentiment"
    ]
    , 
    alpha=20.0
)

train_targets = train.pop(target)
train = encoder.fit_transform(train, train_targets)
test = encoder.transform(test)


train


test


def fill_numeric_with_mean(df: cudf.DataFrame) -> cudf.DataFrame:
    """
    Fill missing values in numeric columns with the column mean.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    return df

def factorize_categoricals(df: cudf.DataFrame) -> cudf.DataFrame:
    """
    Convert remaining object columns to categorical codes (label encoding).
    """
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    for col in cat_cols:
        # factorize returns (encoded_col, categories), 
        # but we only need the encoded column
        df[col], _ = df[col].factorize()
    
    return df

def drop_unnecessary_columns(df: cudf.DataFrame, drop_cols: list) -> cudf.DataFrame:
    """
    Drop columns that are no longer needed.
    """
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    return df


def scale_numeric_features(
    train: cudf.DataFrame, 
    test: cudf.DataFrame
) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    """
    Scale numeric features using cuML's MinMaxScaler.
    """
    # Identify numeric columns in train
    numeric_cols = train.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    
    # Ensure those columns also exist in test 
    # (and optionally intersect with test columns if there's a mismatch)
    numeric_cols_test = test.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    common_numeric_cols = list(set(numeric_cols).intersection(set(numeric_cols_test)))
    common_numeric_cols.sort()  # just to keep a consistent order

    # Optionally, convert train/test numeric columns to float32 
    # to avoid any type mismatch or unexpected behavior:
    train[common_numeric_cols] = train[common_numeric_cols].astype('float32')
    test[common_numeric_cols]  = test[common_numeric_cols].astype('float32')

    # Create a MinMaxScaler and fit on TRAIN columns only
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    # Fit on train
    train[common_numeric_cols] = scaler.fit_transform(train[common_numeric_cols])
    
    # Transform test
    test[common_numeric_cols] = scaler.transform(test[common_numeric_cols])
    
    return train, test

def preprocess_for_elasticnet(
    df: cudf.DataFrame,
    drop_cols: list = None
) -> (cudf.DataFrame, cudf.Series):
    # 1) Drop columns not needed in modeling
    df = drop_unnecessary_columns(df, drop_cols)
    
    # 2) Fill numeric columns
    df = fill_numeric_with_mean(df)
    
    # 3) Factorize remaining object columns
    df = factorize_categoricals(df)

    df = df.reset_index(drop=True)
    
    return df


end = time.time()
print(f"Total time used for preprocessing: {end - start}")


drop_cols_example = [
    'id',
    'Podcast_Name',
    'Episode_Title',
    'Genre',
    'Publication_Day',
    'Publication_Time',
    'Episode_Sentiment',
    'source'
]

train = preprocess_for_elasticnet(
    train,
    drop_cols=drop_cols_example
)

test = preprocess_for_elasticnet(
    test,
    drop_cols=drop_cols_example
)


train, test = scale_numeric_features(train.copy(), test.copy())


import numpy as np
import pandas as pd

from cuml.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X_train = train.reset_index(drop=True)                 
y_train = train_targets.reset_index(drop=True)       
X_test = test.reset_index(drop=True)                    

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

submission[target] = 0

for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\n=== Fold {fold_idx+1}/{N_FOLDS} ===")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    # Create the ElasticNet model with fixed hyperparameters
    model = ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        normalize=False,
        max_iter=1000
    )
    
    # Train
    model.fit(X_tr, y_tr)
    
    # Validate
    val_preds = model.predict(X_val)
    
    model_preds = model.predict(X_test)
    submission[target] += model_preds

submission[target] /= 5
submission.to_csv("submission.csv", index=False)

print("\nSubmission file created: submission.csv")


