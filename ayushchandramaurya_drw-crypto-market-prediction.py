# import packages
import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)




train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")


train.info()


test.info()


#train = train.reset_index(drop=True)


train.head()


# Drop columns have exactly 1 value
NUNIQUE1=[c for c in train.columns if train[c].nunique()==1]
train.drop(NUNIQUE1,axis=1,inplace=True)
test.drop(NUNIQUE1+['label'],axis=1,inplace=True)


top_features = [
    "X344", "X598", "X137", "X174", "X425", "X612", "X167",
    "X168", "X27", "X422", "X342", "X427", "X532", "X178", "X539",
    "X421", "X341", "X465", "X97", "X603", "X138", "X572",
    "X338", "X95", "X161", "X533", "X271", "X279", "X424",
    "X169", "X283", "X332", "X574", "X28", "X281", "X757",
    "X754", "X445", "X180", "X94", "X88", "X525", "X285",
    "X181", "X429", "X343", "X688", "X692", "X680", "X755",
    "X695", "X345", "X611", "X689", "X387", "X588", "X686",
    "X140", "X530", "X753", "X98", "X24", "X756", "X540",
    "X531", "X340", "X383", "X331", "X385", "X277", "X602",
    "X136", "X586", "X300", "X284", "X91", "X379", "X685", "X177",
    'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'
]



train= train[top_features + ["label"]]
test= test[top_features]


# def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
#     start_mem = df.memory_usage(deep=True).sum() / 1024**2
#     print(f"Memory usage before: {start_mem:.2f} MB")

#     for col in df.columns:
#         col_type = df[col].dtype

#         if col_type == 'float64':
#             try:
#                 df[col] = df[col].astype('float16')
#             except ValueError:
#                 pass  

#         elif col_type == 'int64':
#             min_val = df[col].min()
#             max_val = df[col].max()
#             if min_val >= -128 and max_val <= 127:
#                 df[col] = df[col].astype('int8')
#             else:
#                 # optionally handle other int downcasts (int16, int32)
#                 df[col] = pd.to_numeric(df[col], downcast='integer')

#     end_mem = df.memory_usage(deep=True).sum() / 1024**2
#     print(f"Memory usage after: {end_mem:.2f} MB")
#     print(f"Reduced by {(start_mem - end_mem) / start_mem * 100:.1f}%")

#     return df


# train = reduce_memory_usage(train)
# test = reduce_memory_usage(test)



# def add_interaction_features(df):
#     eps = 1e-6

#     df['bid_ask_spread_ratio'] = df['bid_qty'] / (df['ask_qty'] + eps)
#     df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + eps)
#     df['net_qty'] = df['buy_qty'] - df['sell_qty']
#     df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
#     df['liquidity_per_volume'] = df['total_liquidity'] / (df['volume'] + eps)
#     df['trade_density'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
#     df['volume_per_order'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + eps)

#     return df


# train = add_interaction_features(train)
# test = add_interaction_features(test)



train.head()


import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import gc

# Prepare data
X = train.drop('label', axis=1)
y = train['label']

#Params
xgb_params = {
    "colsample_bylevel": 0.477,
    "colsample_bynode": 0.362,
    "colsample_bytree": 0.710,
    "gamma": 1.709,
    "learning_rate": 0.022,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1600,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 39.354,
    "reg_lambda": 65.44,
    "subsample": 0.065, 
    "verbosity": 0
}


# K-Fold setup
split = TimeSeriesSplit(n_splits=14).split(X, y)
xgb_oof_preds = np.zeros(len(X)) 
xgb_scores = []
fold = 1

for train_idx, val_idx in split:
    print(f"Fold {fold}:")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    xgb_oof_preds[val_idx] = preds  # Store OOF predictions

    pearson_corr = np.corrcoef(y_val, preds)[0, 1]
    print(f"Pearson Correlation: {pearson_corr:.5f}")
    xgb_scores.append(pearson_corr)

    del X_train, X_val, y_train, y_val, preds
    gc.collect()

    fold += 1

print(f"\nAverage Pearson Correlation: {np.mean(xgb_scores):.5f}")


final_preds = model.predict(test)
submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
submission["prediction"] = final_preds
submission.to_csv("submission.csv", index=False)
submission.head()




