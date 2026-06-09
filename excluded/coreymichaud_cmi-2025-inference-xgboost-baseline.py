# Libraries
import os
import pandas as pd
import polars as pl
import numpy as np
import joblib
import kaggle_evaluation.cmi_inference_server


# Loading model pipeline
#pipe = joblib.load('/kaggle/input/cmi-2025-training/pipeline.pkl')
pipe = joblib.load('/kaggle/input/testing-xgb-with-wandb/pipeline.pkl')


# Loading label encoder
#le = joblib.load('/kaggle/input/cmi-2025-training/label_encoder.pkl')
le = joblib.load('/kaggle/input/testing-xgb-with-wandb/label_encoder.pkl')


# Getting orientation, thermopile, and time-of-flight column names
acc_cols = [f'acc_{axis}' for axis in ['x', 'y', 'z']]
rot_cols = [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']]
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]


# Function for merging sequence and demographics, creating summary statistics, and grouping by sequence_id
def seq_to_row(sequence, demographics):
    
    # Merging sequence and demographics data
    df = pd.merge(sequence, demographics, on = "subject", how = "left")

    # Getting column names of sequence_id plus the sensor data
    cols = ['sequence_id'] + acc_cols + rot_cols + thm_cols + tof_cols

    # Getting summary statistics
    summary = (df[cols].groupby('sequence_id').agg(['mean', 'std', 'min', 'max', 'median']))

    # Flattening rows
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    # Making sequence_id into a row again
    summary = summary.reset_index()

    # Getting remaining rows
    remaining = df.drop_duplicates('sequence_id')

    # Merging the summary statistics with the other columns
    df = pd.merge(remaining, summary, on = "sequence_id", how = "left")

    return df


# Prediction
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:

    # Converting the parameters to pandas
    seq_df = sequence.to_pandas()
    dem_df = demographics.to_pandas()

    # Merging datasets and converting sequences into flattened rows
    X = seq_to_row(seq_df, dem_df)

    # Predicting
    y_pred = pipe.predict(X)

    # Converting label encoding back into a string
    pred = le.inverse_transform(y_pred)[0]
    
    return pred


# Connecting to evaluation API
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

