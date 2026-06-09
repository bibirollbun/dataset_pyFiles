import polars as pl # Lazy read/preprocess
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import Ridge # Ridge
###############
import os
import warnings
warnings.filterwarnings('ignore')
#############################
# import kaggle_evaluation.jane_street_inference_server # for submission


# R2 wtd

def r2_val(y_true, y_pred, sample_weight):
    nom = np.average((y_pred - y_true) ** 2, weights=sample_weight)
    denom = (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    r2 = 1 -  nom/denom 
    return r2

# Configuration

class CFG: 
    # Note: this is convenient for 
    # updating data/versioning due to different input
    # which is a very common use in Kaggle community
    seed = 42
    target_col = "responder_6"
    feature_cols = ["symbol_id", "time_id"] \
        + [f"feature_{idx:02d}" for idx in range(79)] \
        + [f"responder_{idx}_lag_1" for idx in range(9)] # use the lag data
    categorical_cols = []

# 


DT_GT = 1350 
# Note: Kaggle doesn't have enough RAM for full-size data
# Full-size can be done using chunk-wise run in Colab Pro+

train = pl.scan_parquet( # use lag-1 dataset from a kaggler; not inventing the wheels
    "/kaggle/input/js24-preprocessing-create-lags/training.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

valid = pl.scan_parquet(
    "/kaggle/input/js24-preprocessing-create-lags/validation.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

# train.shape, valid.shape
train = pd.concat([train, valid]).reset_index(drop=True)
train = train.fillna(method = 'ffill').fillna(0)
valid = valid.fillna(method = 'ffill').fillna(0)

# Train vs Valid

X_train = train[ CFG.feature_cols ]
y_train = train[ CFG.target_col ]
w_train = train[ "weight" ]
X_valid = valid[ CFG.feature_cols ]
y_valid = valid[ CFG.target_col ]
w_valid = valid[ "weight" ]

(X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape)


# TRAIN = False
TRAIN = True

if TRAIN:
    model = Ridge() # use the default
    model.fit(X_train,y_train)
    #############################
    train_pred, valid_pred = model.predict(X_train), model.predict(X_valid)
    #############################
    r2_train = r2_val(y_train, train_pred, w_train)
    r2_validate = r2_val(y_valid, valid_pred, w_valid)
    ##############################
    print(f"Train R2: {r2_train}, Validation R2: {r2_validate}")
    # joblib.dump(model, "js24_ridge-base.pkl")

