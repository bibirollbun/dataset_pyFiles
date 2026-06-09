import os
os.system('pip install --force-reinstall /kaggle/input/janestreet2025-code/janestreet-0.1-py3-none-any.whl')

import time
import copy

import numpy as np
import pandas as pd
import polars as pl
import torch 

from kaggle_evaluation import jane_street_inference_server

from janestreet.pipeline import FullPipeline, PipelineCV
from janestreet.data_processor import DataProcessor
from janestreet.config import PATH_DATA


RUN_NAME = "full"
MODEL_NAMES = ["gru_2.0_700", "gru_2.1_700", "gru_2.2_700", "gru_3.0_700", "gru_3.1_700", "gru_3.2_700"]
WEIGHTS = np.array([1.0]*len(MODEL_NAMES))/ len(MODEL_NAMES)
WEIGHTS = WEIGHTS/sum(WEIGHTS)
N_ROLL = 1000

print("_".join(MODEL_NAMES))
print(WEIGHTS)


data_processor = DataProcessor(MODEL_NAMES[0]).load()

pipelines = {}
for model_name in MODEL_NAMES:
    pipeline = FullPipeline(
        None,
        run_name=RUN_NAME,
        name=model_name,
        load_model=True,
        features=None,
        save_to_disc=False
    )
    pipeline.fit(verbose=True)
    pipelines[model_name] = pipeline

    print("-"*100)
    print(model_name)
    print(pipeline.model.get_params())
    print(f"Number of features: {len(pipeline.features)}")
    print(pipeline.model.model.num_resp)


MAX_DATE = 1698
COLS_ID = ['row_id', 'date_id', 'time_id', 'symbol_id', 'weight', 'is_scored']

df_raw = pl.scan_parquet(f"{PATH_DATA}/train.parquet")
df_raw = df_raw.filter(pl.col("date_id")>=MAX_DATE-10)
df_raw = df_raw.collect()
df_raw = df_raw.with_columns(
    pl.lit(-1).cast(pl.Int64).alias("row_id"),
    pl.lit(True).alias("is_scored"),
    (pl.col("date_id")-MAX_DATE-1).alias("date_id")
)
df_raw = df_raw.select(COLS_ID + data_processor.COLS_FEATURES_INIT)

df_raw = (
    df_raw.filter(pl.col("date_id") >= -5)
    .sort(['date_id', 'time_id', 'symbol_id'])
)

hidden_states = [None] * len(pipelines)
dfs = []


import kaggle_evaluation.jane_street_inference_server as jsi

def fixed_run_local_gateway(self, data_paths=None, file_share_dir=None, *args, **kwargs):
    self.server.start()
    try:
        # Call _get_gateway_for_test with ONLY data_paths to avoid argument error
        self.gateway = self._get_gateway_for_test(data_paths)
        self.gateway.run()
    except Exception as err:
        raise err from None
    finally:
        self.server.stop(0)

jsi.JSInferenceServer.run_local_gateway = fixed_run_local_gateway



DEBUG = False

CNT_DATES = 9
CNT_DATES_NOT_SCORED = 4

time_start = time.time()
time_start_not_scored = time.time()
time_est = 0
time_est_not_scored = 0
cnt_dates = 0

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    start_time = time.time()
    
    global df_raw
    global hidden_states
    global pipeline
    global dfs
    global time_est, time_start, time_start_not_scored, time_est_not_scored
    global cnt_dates
    
    date_id = test["date_id"][0]
    time_id = test["time_id"][0]
    is_scored = test["is_scored"][0]

    # Count time for debug
    if DEBUG:
        if not is_scored:
            time_est_not_scored = time.time()-time_start_not_scored
        else:
            time_est = time.time()-time_start

        if time_id == 0:
            print("-" * 100)
            
            if date_id == 1:
                time_start_not_scored = time.time()
    
            if date_id == CNT_DATES_NOT_SCORED: 
                time_start = time.time()

    # Reset hidden states and collect data for weights update
    if time_id == 0:
        cnt_dates += 1
        hidden_states = [None for _, p in pipelines.items()]
        lags = lags.with_columns(
            pl.col("responder_6_lag_1").alias("responder_6"),
            pl.lit(date_id-1).cast(pl.Int16).alias("date_id")
        ).select(["date_id", "time_id", "symbol_id", "responder_6"])
        if cnt_dates > 1:
            df = pl.concat(dfs)
            dfs = []
            df = df.join(lags, on=["date_id", "time_id", "symbol_id"], how="left")
            df = df.sort(["date_id", "time_id", "symbol_id"])

    # Add data to raw dataframe
    test = test.select(df_raw.columns)
    df_raw = pl.concat([df_raw, test], how="vertical_relaxed")
    df_raw = df_raw.select(test.columns)
    
    # Cut raw data (keep last N_ROLL time_ids for each symbol)
    df_raw = (
        df_raw
        .group_by(["symbol_id"])
        .tail(N_ROLL)
    )

    # Calculate features and save
    df_cur = data_processor.process_test_data(df_raw, fast=True, date_id=date_id, time_id=time_id, symbols=test["symbol_id"])
    df_cur = df_cur.sort(["symbol_id"])
    dfs.append(df_cur)
    df_cur = df_cur.with_columns(pl.lit(None).alias("responder_6"))
    
    # Update model weights
    if (time_id == 0) & (cnt_dates > 1):
        if len(df) > 968:
            for i, (name, pipeline) in enumerate(pipelines.items()):
                pipeline.update(df)

    # Make predictions
    if is_scored:
        preds = []
        for i, (name, pipeline) in enumerate(pipelines.items()):
            pred, hidden_states[i] = pipeline.predict(df_cur, hidden=hidden_states[i], n_times=1)
            preds.append(pred)
        pred = np.average(preds, axis=0, weights=WEIGHTS)
    
        df_cur = df_cur.with_columns(pl.Series("responder_6", pred))
        df_cur = test.select(["date_id", "time_id", "symbol_id"]).join(df_cur, on=["date_id", "time_id", "symbol_id"], how="left")
        predictions = df_cur.select(["row_id", "responder_6"])
    else:
        predictions = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )

    if DEBUG:
        if time_id % 100 == 0:
            n_nans = sum(sum(predictions.fill_nan(None).null_count().to_numpy()))
            print(
                f"{date_id} {time_id:3.0f} (is_scored {is_scored}): "
                f"time elps {time.time()-start_time:.4f}, # nans {n_nans}"
            )
    else:
        if (time_id==0)&(date_id==0):
            print(predictions)
            print((time_id, time.time()-start_time))
    
    return predictions

    
inference_server = jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    if not DEBUG:
        inference_server.run_local_gateway(
            [f'{PATH_DATA}/test.parquet', f'{PATH_DATA}/lags.parquet']
        )

    else:
        inference_server.run_local_gateway(
            [
                '/kaggle/input/js24-rmf-submission-api-debug-with-synthetic-test/synthetic_test.parquet',
                '/kaggle/input/js24-rmf-submission-api-debug-with-synthetic-test/synthetic_lag.parquet',
            ]
        )


if DEBUG:
    time_est_cur = time_est/(CNT_DATES-CNT_DATES_NOT_SCORED)*200/60/60
    time_est_scored = time_est/(CNT_DATES-CNT_DATES_NOT_SCORED)*120/60/60
    time_est_not_scored = time_est_not_scored/(CNT_DATES_NOT_SCORED-1)*240/60/60
    time_est_final = time_est_scored + time_est_not_scored
    print("-"*100)
    print(f"Estimated current time: {time_est_cur:.4f}")
    print(f"Estimated final time (is_score=True): {time_est_scored:.4f}")
    print(f"Estimated final time (is_score=False): {time_est_not_scored:.4f}")
    print(f"Estimated final time: {time_est_final:.4f}")

