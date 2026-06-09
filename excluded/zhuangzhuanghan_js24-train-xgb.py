import os
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import pickle
import joblib # dump model as .pkl
import pandas as pd
import polars as pl
############################
from sklearn.metrics import r2_score
# from lightgbm import LGBMRegressor
# import lightgbm as lgb
from xgboost import XGBRegressor # This NB only focus on 
# from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
##########################
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
#########################
# import kaggle_evaluation.jane_street_inference_server


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

# Get model

def get_model(seed):
    # XGBoost parameters
    # easy to tune using this set-up
    XGB_Params = {
        'learning_rate': 0.05, # common use
        'max_depth': 6, # <= sqrt(n_features)
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 5,
        'random_state': seed,
        'tree_method': 'gpu_hist',
        'device' : 'cuda',
        'n_gpus' : 2, # Must turn GPU t4 x 2 on !!!
    }
    
    XGB_Model = XGBRegressor(**XGB_Params)
    return XGB_Model


# Data loading

DT_GT = 1500 
# Note: Kaggle doesn't have enough RAM for full-size data
# Full-size can be done using chunk-wise run in Colab Pro+

train = pl.scan_parquet(
    "/kaggle/input/js24-preprocessing-create-lags/training.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

valid = pl.scan_parquet(
    "/kaggle/input/js24-preprocessing-create-lags/validation.parquet"
).filter(pl.col("date_id") > DT_GT).collect().to_pandas()

# train.shape, valid.shape
train = pd.concat([train, valid]).reset_index(drop=True)

# Train vs Valid

X_train = train[ CFG.feature_cols ]
y_train = train[ CFG.target_col ]
w_train = train[ "weight" ]
X_valid = valid[ CFG.feature_cols ]
y_valid = valid[ CFG.target_col ]
w_valid = valid[ "weight" ]

(X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape)


# Train
# %%time

TRAIN = False # Change to True if needed

if TRAIN:

    model = get_model(CFG.seed)
    model.fit(X_train, y_train, sample_weight=w_train)
    
    # R2 Score
    y_pred_train = model.predict(X_train)
    train_score = r2_score(y_train, y_pred_train, sample_weight=w_train )
    
    y_pred_valid = model.predict(X_valid)
    valid_score = r2_score(y_valid, y_pred_valid, sample_weight=w_valid )
    
    print(f"Train R2: {train_score}, Validation R2: {valid_score}")
    
    # Save
    # joblib.dump(model, "js24_xgb.pkl") # download from the output after saving


