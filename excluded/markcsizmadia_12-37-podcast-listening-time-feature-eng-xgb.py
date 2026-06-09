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


# https://www.kaggle.com/competitions/playground-series-s5e4/discussion/571034
import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


import pandas as pd
from category_encoders import TargetEncoder as CatTargetEncoder, LeaveOneOutEncoder as CatLeaveOneOutEncoder, QuantileEncoder as CatQuantileEncoder
from pathlib import Path
import pandas as pd
from typing import Optional, List
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tqdm import tqdm
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


dir_data = Path("/kaggle/input") / "playground-series-s5e4"

data = pd.read_csv(
    filepath_or_buffer=dir_data / "train.csv",
    dtype={
        'id': 'int64',
        'Podcast_Name': 'category',
        'Episode_Title': 'category',
        'Episode_Length_minutes': 'float32',
        'Genre': 'category',
        'Host_Popularity_percentage': 'float32',
        'Host_Popularity_percentage': 'float32',
        'Publication_Day': 'category',
        'Publication_Time': 'category',
        'Guest_Popularity_percentage': 'float32',
        'Number_of_Ads': 'float32',
        'Episode_Sentiment': 'category',
        'Listening_Time_minutes': 'float32',
        # 'Listening_Time_minutes': 'float32',
    },
)

data_sub = pd.read_csv(
    filepath_or_buffer=dir_data / "test.csv",
    dtype={
        'id': 'int64',
        'Podcast_Name': 'category',
        'Episode_Title': 'category',
        'Episode_Length_minutes': 'float32',
        'Genre': 'category',
        'Host_Popularity_percentage': 'float32',
        'Host_Popularity_percentage': 'float32',
        'Publication_Day': 'category',
        'Publication_Time': 'category',
        'Guest_Popularity_percentage': 'float32',
        'Number_of_Ads': 'float32',
        'Episode_Sentiment': 'category',
        'Listening_Time_minutes': 'float32',
    },
)


data.duplicated().sum()


comb_features = ["Podcast_Name", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title"]
comb_feature_names = []
for features in tqdm(list(combinations(comb_features, r=2)) + list(combinations(comb_features, r=3)) + list(combinations(comb_features, r=4))):
    feature_name = "_".join(features)
    comb_feature_names.append(feature_name)
    data.loc[:, feature_name] = data.apply(lambda x: "_".join([str(x[feature]) for feature in features]), axis=1).astype("category")
    data_sub.loc[:, feature_name] = data_sub.apply(lambda x: "_".join([str(x[feature]) for feature in features]), axis=1).astype("category") # potentital category diff is fine, we encode with the same preprocessor later
data


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


features_comb = [
    "episode_length__Episode_Length_minutes",
    # "podcast_name__Podcast_Name",
    # "episode_title__Episode_Title",
    # "genre__Genre",
    # "publication_day__Publication_Day",
    # "publication_time__Publication_Time",
    # "episode_sentiment__Episode_Sentiment",
    "number_of_ads__Number_of_Ads",
    "host_popularity_pct__Host_Popularity_percentage",
    "guest_popularity_pct__Guest_Popularity_percentage"
]
print(len(features_comb))
features_comb_2 = list(combinations(features_comb, r=2))
feature_names_comb = \
    features_comb_2 \
    + list(combinations(features_comb, r=3)) \
    + list(combinations(features_comb, r=4)) \
    + list(combinations(features_comb, r=5)) \
    + list(combinations(features_comb, r=6)) \
    + list(combinations(features_comb, r=7)) \
    + list(combinations(features_comb, r=8)) \
    + list(combinations(features_comb, r=9)) \
    + list(combinations(features_comb, r=10))
print(len(feature_names_comb))
print(feature_names_comb[:10])


features_comb_2_v2 = [
    (feature_comb_2[1], feature_comb_2[0]) for feature_comb_2 in features_comb_2
]
features_comb_2_v2


def multiply(X: pd.DataFrame) -> pd.DataFrame:
    cols = X.columns
    col_name = f"{cols[0]}_x_{cols[1]}"
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


def get_preprocessor(random_state: Optional[int] = None) -> Pipeline:
    transformers=[
        ("podcast_name", OrdinalEncoder(categories=[podcast_name_ordered]), ["Podcast_Name"]),
        
        ("episode_title", OrdinalEncoder(categories=[episode_title_ordered]), ["Episode_Title"]),
        
        ("episode_length", "passthrough", ["Episode_Length_minutes"]),
        
        ("genre", OrdinalEncoder(categories=[genre_ordered]), ["Genre"]),
        
        ("host_popularity_pct", "passthrough", ["Host_Popularity_percentage"]),
        
        ("publication_day", OrdinalEncoder(categories=[publication_day_ordered]), ["Publication_Day"]),
        
        ("publication_time", OrdinalEncoder(categories=[publication_time_ordered]), ["Publication_Time"]),
        
        ("guest_popularity_pct", "passthrough", ["Guest_Popularity_percentage"]),
        
        ("number_of_ads", "passthrough", ["Number_of_Ads"]),
        
        ("episode_sentiment", OrdinalEncoder(categories=[episode_sentiment_ordered]), ["Episode_Sentiment"]),

        # ("target_me", CatLeaveOneOutEncoder(cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        ("target_me", CatTargetEncoder(cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),

        ("target_q5", CatQuantileEncoder(cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        ("target_q25", CatQuantileEncoder(quantile=0.25, cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        ("target_q75", CatQuantileEncoder(quantile=0.25, cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        ("target_q10", CatQuantileEncoder(quantile=0.10, cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        ("target_q90", CatQuantileEncoder(quantile=0.90, cols=["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]), ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]),
        
        ("comb_features", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), comb_feature_names),
    ]
    combinators = [
        ("pass", "passthrough", features_comb),
    ] + [
        (f"cx_{i}", FunctionTransformer(func=multiply), f) for i, f in enumerate(feature_names_comb)
    ] + [
        (f"c1/_{i}", FunctionTransformer(func=divide), f) for i, f in enumerate(features_comb_2)
    ] + [
        (f"c2/_{i}", FunctionTransformer(func=divide), f) for i, f in enumerate(features_comb_2_v2)
    ]
    preprocessor = Pipeline(
        steps=[
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("combinator", ColumnTransformer(transformers=combinators, remainder="passthrough"))
        ]
    )
    return preprocessor


random_state = 123
X = data.drop(columns=["Listening_Time_minutes"])
y = data["Listening_Time_minutes"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=random_state, stratify=X["Podcast_Name_Episode_Title"])


%%time
preprocessor = get_preprocessor(random_state=random_state)
X_train_val_p = preprocessor.fit_transform(X=X_train_val, y=y_train_val)
X_test_p = preprocessor.transform(X=X_test)


print(f"Using {len(X_test_p.columns)} features")


def get_model(random_state: Optional[int] = None):
    clf = XGBRegressor(
        learning_rate=0.01,
        n_estimators=10_000,
        colsample_bytree=0.9,
        subsample=0.9,
        max_depth=16,
        random_state=random_state,
        early_stopping_rounds=25,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda"
        )
    return clf


import time

def k_fold(data_train: pd.DataFrame, data_test: pd.DataFrame, n_folds: int = 5, random_state: Optional[int] = None):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof = np.zeros(data_train.shape[0])
    y_test_hat_ave = np.zeros(data_test.shape[0])

    X = data.drop(columns=["Listening_Time_minutes"])
    y = data["Listening_Time_minutes"]
    X_test = data_test.copy()
    iter = tqdm(enumerate(kf.split(X, y=X["Podcast_Name_Episode_Title"])), total=n_folds)
    for i, (train_index, valid_index) in iter:
        
        X_train: pd.DataFrame = X.loc[train_index]
        y_train: pd.DataFrame = y.loc[train_index]
        X_valid: pd.DataFrame = X.loc[valid_index]
        y_valid: pd.DataFrame = y.loc[valid_index]

        preprocessor = get_preprocessor(random_state=random_state)
        X_train_p: pd.DataFrame = preprocessor.fit_transform(X=X_train, y=y_train)
        X_valid_p: pd.DataFrame = preprocessor.transform(X=X_valid)
        X_test_p: pd.DataFrame = preprocessor.transform(X=X_test)

        clf = get_model(random_state=random_state)
        clf.fit(X=X_train_p, y=y_train, eval_set=[(X_valid_p, y_valid)], verbose=10)
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


y_test_hat_ave, oof = k_fold(data_train=data, data_test=data_sub, n_folds=20, random_state=random_state)


local_cv_score = np.sqrt(mean_squared_error(y_true=data["Listening_Time_minutes"], y_pred=oof))
print(f"Overall CV RMSE = {local_cv_score:.5f}")


data_sub["Listening_Time_minutes"] = y_test_hat_ave
data_sub[["id", "Listening_Time_minutes"]].to_csv("submission.csv", index=False)




