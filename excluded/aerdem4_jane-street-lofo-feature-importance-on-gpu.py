!pip install -q lofo-importance


import numpy as np
import pandas as pd
import os, sys, gc
from tqdm import tqdm
import polars as pl


df = pl.scan_parquet(
    f"/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet"
).filter(
    pl.col("date_id").gt(1000)
).collect().sample(fraction=0.05, seed=0).to_pandas()

print(df.shape)

gc.collect()


anon_features = [f"feature_{str(i).zfill(2)}" for i in range(79)]
feature_cols = ["weight"] + anon_features + ["time_id"]
target = "responder_6"

len(feature_cols)


VAL_LEN = 65

max_date = df["date_id"].max()

val_scheme = []

for i in range(1, 6):
    train_ind = np.where(df["date_id"].values < max_date - VAL_LEN*i)[0]
    val_ind = np.where((df["date_id"].values >= max_date - VAL_LEN*i) & 
                       (df["date_id"].values <= max_date - VAL_LEN*(i-1)))[0]
    
    print(len(train_ind), len(val_ind))
    
    val_scheme.append((train_ind, val_ind))


import lofo

ds = lofo.Dataset(df, target=target, features=feature_cols, auto_group_threshold=0.7)


from xgboost import XGBRegressor

xgb_param = {
        'learning_rate': 0.04,
        'max_depth': 5,
        'colsample_bynode': 0.6,
        'reg_alpha': 1,
        'reg_lambda': 5,
        'random_state': 0,
        'device' : 'cuda',
    "min_child_weight": 128,
    "n_estimators":100
    }


model = XGBRegressor(
    **xgb_param
)


lofo_imp = lofo.LOFOImportance(ds, cv=val_scheme, scoring="neg_mean_squared_error", 
                               model=model, n_jobs=1, fit_params={'sample_weight': df["weight"]})
imp_df = lofo_imp.get_importance()


lofo.plot_importance(imp_df, figsize=(12, 18))


imp_df.to_csv("importance.csv", index=False)

