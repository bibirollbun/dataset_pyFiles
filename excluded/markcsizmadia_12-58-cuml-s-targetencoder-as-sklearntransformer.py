# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
from category_encoders import TargetEncoder as CatTargetEncoder, LeaveOneOutEncoder as CatLeaveOneOutEncoder, QuantileEncoder as CatQuantileEncoder
from pathlib import Path
import pandas as pd
from typing import Optional, List
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tqdm import tqdm
from cuml.preprocessing import TargetEncoder
from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn import set_config
set_config(transform_output="pandas")


import pandas as pd
import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


dir_data = Path("/kaggle/input") / "playground-series-s5e4"
path_data_original = Path("/kaggle/input") / "podcast-listening-time-prediction-dataset" / "podcast_dataset.csv"

data_syn = pd.read_csv(
    filepath_or_buffer=dir_data / "train.csv",
    dtype={
        'id': 'int64',
        'Podcast_Name': 'str',
        'Episode_Title': 'str',
        'Episode_Length_minutes': 'float32',
        'Genre': 'str',
        'Host_Popularity_percentage': 'float32',
        'Host_Popularity_percentage': 'float32',
        'Publication_Day': 'str',
        'Publication_Time': 'str',
        'Guest_Popularity_percentage': 'float32',
        'Number_of_Ads': 'float32',
        'Episode_Sentiment': 'str',
        'Listening_Time_minutes': 'float32',
    },
)

data_original = pd.read_csv(
    filepath_or_buffer=path_data_original,
    dtype={
        'id': 'int64',
        'Podcast_Name': 'str',
        'Episode_Title': 'str',
        'Episode_Length_minutes': 'float32',
        'Genre': 'str',
        'Host_Popularity_percentage': 'float32',
        'Host_Popularity_percentage': 'float32',
        'Publication_Day': 'str',
        'Publication_Time': 'str',
        'Guest_Popularity_percentage': 'float32',
        'Number_of_Ads': 'float32',
        'Episode_Sentiment': 'str',
        'Listening_Time_minutes': 'float32',
    },
)

data_sub = pd.read_csv(
    filepath_or_buffer=dir_data / "test.csv",
    dtype={
        'id': 'int64',
        'Podcast_Name': 'str',
        'Episode_Title': 'str',
        'Episode_Length_minutes': 'float32',
        'Genre': 'str',
        'Host_Popularity_percentage': 'float32',
        'Publication_Day': 'str',
        'Publication_Time': 'str',
        'Guest_Popularity_percentage': 'float32',
        'Number_of_Ads': 'float32',
        'Episode_Sentiment': 'str',
        'Listening_Time_minutes': 'float32',
    },
)


print(f"Original data shape before deduplication: {data_original.shape[0]}")
data_original = data_original[~data_original.duplicated()]
print(f"Original data shape after deduplication: {data_original.shape[0]}")
data_original = data_original[data_original["Listening_Time_minutes"].notnull()]
print(f"Original data shape after dropping NaN targets: {data_original.shape[0]}")


print(f"Synthetic data shape before deduplication: {data_syn.shape[0]}")
data_syn = data_syn[~data_syn.duplicated()]
print(f"Synthetic data shape after deduplication: {data_syn.shape[0]}")


data = pd.concat(objs=[data_syn, data_original], axis=0, ignore_index=True)
print(f"Data shape: {data.shape[0]}")


random_state = 123
encoder = TargetEncoder(split_method="interleaved", smooth=20, n_folds=10, seed=random_state)
data["TE_Podcast_Name"] = encoder.fit_transform(data["Podcast_Name"], data["Listening_Time_minutes"])
data[["TE_Podcast_Name", "Podcast_Name", "Listening_Time_minutes"]]


class MyTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, stat: str, cols: List[str], random_state: int, split_method: str, smooth: int, n_folds: int):
        self.stat = stat
        self.cols = cols
        self.random_state = random_state
        self.split_method = split_method
        self.smooth = smooth
        self.n_folds = n_folds
        self.te_map = {col: TargetEncoder(stat=self.stat, seed=self.random_state, split_method=split_method, smooth=smooth, n_folds=n_folds,) for col in self.cols}
        self.if_fit = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        columns = [col for col in X.columns if col in self.cols]
        for col in columns:
            self.te_map[col].fit(X[col], y)
        return self
        
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        columns = [col for col in X.columns if col in self.cols]
        te_columns = [f"te_{col}" for col in columns]
        for col, te_col in zip(columns, te_columns):
            X[te_col] = self.te_map[col].transform(X[col])
        return X[te_columns]


myenc = MyTargetEncoder(stat="mean", cols=["Genre", "Podcast_Name"], random_state=123, split_method="interleaved", smooth=20, n_folds=10)
myenc.fit(data, data["Listening_Time_minutes"])
myenc.transform(data, data["Listening_Time_minutes"])


data.info()


primary_numerical_features_comb2 = list(combinations(["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"], r=2))
primary_numerical_features_comb2_inv = [
    (feature_comb_2[1], feature_comb_2[0]) for feature_comb_2 in primary_numerical_features_comb2
]
primary_numerical_features_comb2_inv


coly = "Listening_Time_minutes"
colx = "Podcast_Name"
podcast_name_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
colx = "Episode_Title"
episode_title_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
colx = "Genre"
genre_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
colx = "Publication_Day"
publication_day_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
colx = "Publication_Time"
publication_time_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
colx = "Episode_Sentiment"
episode_sentiment_ordered = data.groupby(colx, observed=True)[coly].agg(["mean"]).sort_values(by="mean", ascending=True).reset_index()[colx].tolist()
publication_time_ordered


def multiply(X: pd.DataFrame) -> pd.DataFrame:
    col_name = "_x_".join(X.columns)
    X[col_name] = X.product(axis=1)
    return X[[col_name]]


def divide(X: pd.DataFrame) -> pd.DataFrame:
    eps = 10e-6
    cols = X.columns
    col_name = f"{cols[0]}_/_{cols[1]}"
    assert len(cols) == 2
    X[col_name] = X.iloc[:, 0] / (X.iloc[:, 1] + eps)
    return X[[col_name]]


def subtract(X: pd.DataFrame) -> pd.DataFrame:
    cols = X.columns
    col_name = f"{cols[0]}_-_{cols[1]}"
    assert len(cols) == 2
    X[col_name] = X.iloc[:, 0] - X.iloc[:, 1]
    return X[[col_name]]


def get_preprocessor(random_state: Optional[int] = None) -> Pipeline:
    transformers=[
        ("podcast_name", OrdinalEncoder(categories=[podcast_name_ordered]), ["Podcast_Name"]),
        ("episode_title", OrdinalEncoder(categories=[episode_title_ordered]), ["Episode_Title"]),
        ("genre", OrdinalEncoder(categories=[genre_ordered]), ["Genre"]),
        ("publication_day", OrdinalEncoder(categories=[publication_day_ordered]), ["Publication_Day"]),
        ("publication_time", OrdinalEncoder(categories=[publication_time_ordered]), ["Publication_Time"]),
        ("episode_sentiment", OrdinalEncoder(categories=[episode_sentiment_ordered]), ["Episode_Sentiment"]),
        ("primary_nums", "passthrough", ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]),
        ("ce1", MyTargetEncoder(stat="mean", cols=["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"], random_state=random_state, split_method="interleaved", smooth=20, n_folds=10), ["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"]),
        ("ce2", MyTargetEncoder(stat="median", cols=["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"], random_state=random_state, split_method="interleaved", smooth=20, n_folds=10), ["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"]),
        ("ce3", MyTargetEncoder(stat="var", cols=["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"], random_state=random_state, split_method="interleaved", smooth=20, n_folds=10), ["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title", "Genre"]),
    ] + [
        (f"c1/_{i}", FunctionTransformer(func=divide), f) for i, f in enumerate(primary_numerical_features_comb2)
    ] + [
        (f"c2/_{i}", FunctionTransformer(func=divide), f) for i, f in enumerate(primary_numerical_features_comb2_inv)
    ]
    preprocessor = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
        ]
    )
    return preprocessor


random_state = 123
X = data.drop(columns=["Listening_Time_minutes"])
y = data["Listening_Time_minutes"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=random_state)


preprocessor = get_preprocessor(random_state=random_state)
X_train_val_p = preprocessor.fit_transform(X=X_train_val, y=y_train_val)
X_test_p = preprocessor.transform(X=X_test)


X_train_val_p.columns


def get_model(random_state: Optional[int] = None):
    def lr_decay(epoch):
        if epoch < 1_000:
            lr = 0.03
        else:
            lr = 0.015
        return lr
        
    callback = xgb.callback.LearningRateScheduler(lr_decay)
    clf = xgb.XGBRegressor(
        n_estimators=10_000,
        colsample_bytree=0.8,
        subsample=0.9,
        reg_lambda=8,
        max_depth=12,
        random_state=random_state,
        early_stopping_rounds=25,
        callbacks=[callback],
        eval_metric="rmse",
        tree_method="hist",
        device="cuda:0"
        )
    return clf


def k_fold(data_train: pd.DataFrame, data_test: pd.DataFrame, n_folds: int = 5, random_state: Optional[int] = None):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(data_train.shape[0])
    y_test_hat_ave = np.zeros(data_test.shape[0])

    X = data.drop(columns=["Listening_Time_minutes"])
    y = data["Listening_Time_minutes"]
    X_test = data_test.copy()
    iter = tqdm(enumerate(kf.split(X)), total=n_folds)
    for i, (train_index, valid_index) in iter:
        
        X_train: pd.DataFrame = X.loc[train_index]
        y_train: pd.DataFrame = y.loc[train_index]
        y_train += np.random.normal(size=y_train.shape[0]) # add noise to target (a form of regularisation)
        X_valid: pd.DataFrame = X.loc[valid_index]
        y_valid: pd.DataFrame = y.loc[valid_index]

        preprocessor = get_preprocessor(random_state=random_state)
        X_train_p: pd.DataFrame = preprocessor.fit_transform(X=X_train, y=y_train)
        X_valid_p: pd.DataFrame = preprocessor.transform(X=X_valid)
        X_test_p: pd.DataFrame = preprocessor.transform(X=X_test)

        clf = get_model(random_state=random_state)
        clf.fit(X=X_train_p, y=y_train, eval_set=[(X_train_p, y_train), (X_valid_p, y_valid)], verbose=100)
        y_valid_hat = clf.predict(X=X_valid_p)
        y_test_hat = clf.predict(X=X_test_p)
        # INFER OOF
        oof[valid_index] = y_valid_hat
        # INFER TEST
        y_test_hat_ave += y_test_hat

        current_cv_score = np.sqrt(mean_squared_error(y_true=y_valid.to_numpy(), y_pred=oof[valid_index]))
        iter.set_description(f" => Fold {i+1} RMSE = {current_cv_score:.5f}")

    # average tets preds (bagging)
    y_test_hat_ave /= n_folds

    return y_test_hat_ave, oof


random_state = 123
y_test_hat_ave, oof = k_fold(data_train=data, data_test=data_sub, n_folds=10, random_state=random_state)


local_cv_score = np.sqrt(mean_squared_error(y_true=data["Listening_Time_minutes"], y_pred=oof))
print(f"Overall CV RMSE = {local_cv_score:.5f}")


data_sub["Listening_Time_minutes"] = y_test_hat_ave
data_sub[["id", "Listening_Time_minutes"]].to_csv("submission.csv", index=False)




