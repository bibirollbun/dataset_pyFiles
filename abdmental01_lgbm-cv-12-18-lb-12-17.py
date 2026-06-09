%%time

!pip install -qq scikit-learn==1.6.1


%%time

from tqdm import tqdm
from itertools import combinations
from xgboost import XGBRegressor


import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

import lightgbm as lgb

from cuml.preprocessing import TargetEncoder
import cudf
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore')


%%time

def feature_eng(df):
    
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    df = df.drop(columns=['Episode_Title'])
    return df


%%time

df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
df_train = feature_eng(df_train)

df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
df_test = feature_eng(df_test)

df_subm = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')

cat_c = ['Episode_Num', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment','Podcast_Name','Genre']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')
    return df

df_train = update(df_train)
df_test = update(df_test)


%%time

def n_fe(df):
    import numpy as np
    
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    
    return df

df_train = n_fe(df_train)
df_test = n_fe(df_test)


%%time

encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4, 5]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        df_train[new_col_name] = df_train[list(cols)].astype(str).agg('_'.join, axis=1)
        df_train[new_col_name] = df_train[new_col_name].astype('category')
        
        df_test[new_col_name] = df_test[list(cols)].astype(str).agg('_'.join, axis=1)
        df_test[new_col_name] = df_test[new_col_name].astype('category')


%%time

X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']


%%time

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def TRAIN(params, encoded_column_start_index=11, n_splits=7):
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    y_pred = np.zeros(len(df_subm))
    oof = np.zeros(len(X))
    rmse_scores = []

    for fold, (idx_train, idx_valid) in enumerate(cv.split(X, y)):
        print(f"\nðŸ“¦ Fold {fold + 1}")

        X_train = cudf.from_pandas(X.iloc[idx_train].copy())
        X_valid = cudf.from_pandas(X.iloc[idx_valid].copy())
        X_test  = cudf.from_pandas(df_test[X.columns].copy())
        y_train = cudf.Series(y.iloc[idx_train].copy())
        y_valid = y.iloc[idx_valid].copy()

        encoded_columns = X.columns[encoded_column_start_index:]
        print("ðŸŽ¯ Target encoding: ", end="")
        for c in tqdm(encoded_columns, desc="Encoding columns"):
            encoder = TargetEncoder(
                n_folds=5,
                smooth=0,
                split_method='random',
                stat='mean'
            )
            X_train[c] = encoder.fit_transform(X_train[[c]], y_train)
            X_valid[c] = encoder.transform(X_valid[[c]])
            X_test[c]  = encoder.transform(X_test[[c]])

        X_train = X_train.to_pandas()
        X_valid = X_valid.to_pandas()
        X_test  = X_test.to_pandas()
        y_train = y_train.to_pandas()

        model = LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                early_stopping(stopping_rounds=700),
                log_evaluation(period=100)
            ]
        )

        oof[idx_valid] = model.predict(X_valid)
        y_pred += model.predict(X_test)

        fold_rmse = mean_squared_error(y_valid, oof[idx_valid]) ** 0.5
        rmse_scores.append(fold_rmse)
        print(f"âœ… Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    overall_rmse = mean_squared_error(y, oof) ** 0.5
    print(f"\nðŸŽ¯ Overall CV RMSE: {overall_rmse:.5f}")

    y_pred /= n_splits
    return y_pred, oof, overall_rmse


%%time

params_lgbm = {
    'n_estimators': 50000,
    'learning_rate': 0.00462847749422193,
    'max_depth': 10,
    'min_data_in_leaf': 4,
    'subsample': 0.8244,
    'colsample_bytree': 0.5586,
    'reg_lambda': 0.3548,
    'boosting_type': 'gbdt',
    'objective': 'rmse',
    'random_state': 2025,
    'device': 'gpu'
}


y_pred_lgb, oof, overall_rmse = TRAIN(params_lgbm)


%%time

df_subm['Listening_Time_minutes'] = y_pred_lgb
df_subm.to_csv('submission_CAT_1.csv')
df_subm.head()

