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
import polars as pl
import numpy as np
import os
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import pickle

from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None




train = pl.read_parquet(f"/kaggle/input/training/training.parquet")


valid = pl.read_parquet(f"/kaggle/input/training/validation.parquet")


n_skip = 25  # Number of last rows to skip
filtered_train = train[:-n_skip]


filtered_train.shape, valid.shape


import kaggle_evaluation.jane_street_inference_server




class CONFIG:
    seed = 42
    target_col = "responder_6"
    feature_cols = [col for col in valid.columns if ('feature' in col) or ('_lag_' in col)]
    
        
    


import os
from xgboost import XGBRegressor

# Fit the model
X_train = train[ CONFIG.feature_cols ]
y_train = train[ CONFIG.target_col ]
w_train = train[ "weight" ]
X_valid = valid[ CONFIG.feature_cols ]
y_valid = valid[ CONFIG.target_col ]
w_valid = valid[ "weight" ]

# Define the model globally
XGB_Model = None

def get_model(seed, X_train, y_train, w_train):
    # Define the seed
    seed = 42  # or any other integer value
    
    # XGBoost parameters
    model = XGBRegressor(
        learning_rate=0.05,
        max_depth=9,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=5,
        random_state=seed,
        tree_method='hist',
        device='cuda',
    )

    # Fit the model
    X_train = train[ CONFIG.feature_cols ]
    y_train = train[ CONFIG.target_col ]
    w_train = train[ "weight" ]
    X_valid = valid[ CONFIG.feature_cols ]
    y_valid = valid[ CONFIG.target_col ]
    w_valid = valid[ "weight" ]

    # Fit the model with sample weights
    model.fit(X_train, y_train, sample_weight=w_train)

    
    
    return model


X_train.shape, y_train.shape, w_train.shape, X_valid.shape, y_valid.shape, w_valid.shape


X_valid = valid[ CONFIG.feature_cols ]
y_valid = valid[ CONFIG.target_col ]
w_valid = valid[ "weight" ]

X_valid.shape, y_valid.shape, w_valid.shape


import pandas as pd
import polars as pl

lags_: pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    test_pd = test.to_pandas()
    
    if not lags is None:
        lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
        test = test.join(lags, on=["date_id", "symbol_id"],  how="left")
    else:
        test = test.with_columns(
            ( pl.lit(0.0).alias(f'responder_{idx}_lag_1') for idx in range(9) )
        ) 

    feature_cols = [col for col in test_pd.columns if ('feature' in col) or ('responder' in col)]
    print("Feature columns for prediction:", feature_cols)  
    
    # Prepare the features for prediction
    test_pd = test_pd.fillna(0)
    print("Number of NaN values after filling:", test_pd.isna().sum().sum())

    model = get_model(CONFIG.seed, X_train, y_train, w_train)
    # Make predictions using the XGB_Model
    predictions = predict(test_pd[feature_cols])

    # Create a DataFrame for the output
    output = pd.DataFrame({
        'row_id': test_pd['row_id'],
        'responder_6': predictions
    })

    # Ensure the output DataFrame has the correct format
    assert output.columns.tolist() == ['row_id', 'responder_6']
    assert len(output) == len(test)

    return output


import kaggle_evaluation.jane_street_inference_server

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )







