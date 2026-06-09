import pandas as pd
import polars as pl
import numpy as np
import os, gc
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import *
import joblib
import kaggle_evaluation.jane_street_inference_server


feature_cols = [f"feature_{idx:02d}" for idx in range(79)]+ [f"responder_{idx}_lag_1" for idx in range(9)]
target_col = "responder_6"
selected_features = ["symbol_id", "time_id"] + feature_cols+[target_col]
features = ["symbol_id", "time_id"] + feature_cols


valid = pl.scan_parquet("/kaggle/input/js-24-dataset-with-lags/validation.parquet").collect().to_pandas()

X_valid = valid[features]
y_valid = valid['responder_6']
w_valid = valid["weight"]


model_lgb = joblib.load("/kaggle/input/js-24-traind-models/lightgbm_model2.pkl")


y_pred_valid = model_lgb.predict(X_valid)
valid_score = r2_score(y_valid,y_pred_valid)
valid_score


lags_ : pl.DataFrame | None = None
    
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    global lags_
    if lags is not None:
        lags_ = lags

    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )
    symbol_ids = test.select('symbol_id').to_numpy()[:, 0]

    lags = lags_.clone().group_by(["date_id", "symbol_id"], maintain_order=True).last() # pick up last record of previous date
    test = test.join(lags, on=["date_id", "symbol_id"],  how="left")

    
    """ Pred LGB """
    preds = model_lgb.predict(test[features].to_pandas())

    """ Finaly """
    predictions = \
    test.select('row_id').\
    with_columns(
        pl.Series(
            name   = 'responder_6', 
            values = np.clip(preds, a_min = -5, a_max = 5),
            dtype  = pl.Float64,
        )
    )

    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    assert list(predictions.columns) == ['row_id', 'responder_6']
    assert len(predictions) == len(test)

    return predictions


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

print('submitted')

