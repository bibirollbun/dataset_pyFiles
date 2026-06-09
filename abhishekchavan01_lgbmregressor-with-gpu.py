%load_ext cuml.accel


import lightgbm as lgb
import warnings
import sklearn
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from itertools import combinations
from category_encoders import TargetEncoder
warnings.simplefilter('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sum = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


le = LabelEncoder()
encoder = TargetEncoder()


def feature_eng(df):
    df['Episode_Num'] = df['Episode_Title'].str[8:]
    cols_to_encode = [
        'Genre', 'Podcast_Name', 
        'Publication_Day', 
        'Publication_Time', 
        'Episode_Sentiment'
    ]
    
    for col in cols_to_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        df[col] = df[col].astype('category')   
    df = df.drop(columns=['Episode_Title'])
    return df



df_train = feature_eng(df_train)
df_test = feature_eng(df_test)


df_train.head()


encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4]
for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        df_train[new_col_name] = df_train[list(cols)].astype(str).agg('_'.join, axis=1)
        df_train[new_col_name] = df_train[new_col_name].astype('category')        
        df_test[new_col_name] = df_test[list(cols)].astype(str).agg('_'.join, axis=1)
        df_test[new_col_name] = df_test[new_col_name].astype('category')


X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']


cv = KFold(5, random_state=42, shuffle=True)
y_pred = np.zeros(len(df_sum))

for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = df_test[X.columns].copy()
    
    encoded_columns = df_train.columns[11:]        
    X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

    model = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(100)],
    )
    y_pred += model.predict(X_test)


df_sum['Listening_Time_minutes'] = y_pred / 5
df_sum.to_csv('submission.csv', index = False)
df_sum.head()


df_sum.info()




