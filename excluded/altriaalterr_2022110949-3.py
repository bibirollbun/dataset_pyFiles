import os
import joblib
import polars as pl
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import kaggle_evaluation.jane_street_inference_server


input_path = '/kaggle/input/jane-street-real-time-market-data-forecasting'
def read_selected_data(input_path):
    selected_files = [f"partition_id={i}/part-0.parquet" for i in range(1)]
    dfs = []
    for file_name in selected_files:
        file_path = f'{input_path}/train.parquet/{file_name}'
        lazy_df = pl.scan_parquet(file_path)
        df = lazy_df.collect()
        dfs.append(df)
    full_df = pl.concat(dfs)
    return full_df


df = read_selected_data(input_path)
df = df.fill_null(strategy='forward')
feature_names = [f"feature_{i:02d}" for i in range(79)]
tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(df.to_pandas()):
    train_dates = df[train_index]['date_id']
    valid_dates = df[val_index]['date_id']


X_valid = df.filter(pl.col('date_id').is_in(valid_dates)).select(feature_names).to_numpy()
y_valid = df.filter(pl.col('date_id').is_in(valid_dates)).select('responder_6').to_numpy().ravel()
w_valid = df.filter(pl.col('date_id').is_in(valid_dates)).select('weight').to_numpy().ravel()
X_train = df.filter(pl.col('date_id').is_in(train_dates)).select(feature_names).to_numpy()
y_train = df.filter(pl.col('date_id').is_in(train_dates)).select('responder_6').to_numpy().ravel()
w_train = df.filter(pl.col('date_id').is_in(train_dates)).select('weight').to_numpy().ravel()
feature_columns = [col for col in df.columns if 'feature' in col]
df = df.with_columns([
    pl.col(col).fill_null(0).alias(col) for col in feature_columns
])


df


def r2_xgb(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return -r2


model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    max_depth=6,
    tree_method='hist',
    objective='reg:squarederror',
    eval_metric=r2_xgb,
    disable_default_eval_metric=True,
    early_stopping_rounds=2
)


model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_valid, y_valid)],
    sample_weight_eval_set=[w_valid]
    )


if not os.path.exists("./model_save"):
    # Create the directory if it does not exist
    os.mkdir("./model_save")
model.save_model('./model_save/2022110949.json')


model_loaded = xgb.XGBRegressor()
model_loaded.load_model('/kaggle/working/model_save/2022110949.json')


test = pl.scan_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet/date_id=0/part-0.parquet")
test = test.collect()
test = test.to_pandas()
test = test[feature_names].values
predictions = model_loaded.predict(test)
predictions


test = pl.scan_parquet("/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet/date_id=0/part-0.parquet")
test = test.collect()
test = test.to_pandas()

test_df = test[feature_names].values
predictions = model_loaded.predict(test_df)

output_df = pd.DataFrame({"row_id": test['row_id'], "responder_6": predictions})


output_df.head()


# Global lags storage
lags_: pl.DataFrame | None = None
def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global lags_, model_loaded # Declare models as global
    
    # Logic for saving or loading lags
    if lags is not None:
        lags_ = lags
    
    test = test.to_pandas()
    test_df = test[feature_names].values
    predictions = model_loaded.predict(test_df)

    output_df = pd.DataFrame({"row_id": test['row_id'], "responder_6": predictions})

        
    return pl.from_pandas(output_df)


# Setup the inference server
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

# Running the inference server
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((
        '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
        '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
    ))

