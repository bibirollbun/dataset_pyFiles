import kaggle_evaluation.cmi_inference_server
import polars as pl
import pandas as pd
import re
import pickle
import os
import numpy as np

def remove_thm_outliers(df, threshold=20):

    # Thermophile outlier fix
    for column in df.columns:
        if "thm" in column:
            mask = df[column] < threshold
            if mask.any():
                mean_val = df.loc[~mask, column].mean()
                df.loc[mask, column] = mean_val
    return df

def impute_tof_values(df):
    for column in df.columns:
        if "tof" in column:
            # Exclude -1 values when calculating the mean
            valid_values = df[column][df[column] != -1]
            mean_val = valid_values.mean(skipna=True)
            max_val = df[column].max(skipna=True)

            # Replace NaN with the mean
            df[column] = df[column].fillna(mean_val)
            # Replace -1 with the maximum value
            df[column] = df[column].replace(-1, max_val)
    return df

def drop_columns(columns, df):

    df = df.drop(columns=columns)
    return df

def average_tof_values(df):

    tof_pattern = re.compile(r"^(tof_\w+)_v(\d+)$")
    tof_groups = {}
    for col in df.columns:
        m = tof_pattern.match(col)
        if m:
            base = m.group(1)
            tof_groups.setdefault(base, []).append(col)

    df_tof_avg = []
    df_tof_avg_name = []
    for base, cols in tof_groups.items():
        
        tof_sensor_average = df[cols].mean(axis=1)
        df_tof_avg.append(tof_sensor_average)
        df_tof_avg_name.append(f"{base}_avg")

    # Drop all columns containing 'tof'
    tof_cols = [col for col in df.columns if "tof" in col]
    df = df.drop(columns=tof_cols)

    for name, avg in zip(df_tof_avg_name, df_tof_avg):
        df[name] = avg
    return df

### Hard-coded invertet gesture
inv_gesture_numerical_dict = {1: 'Cheek - pinch skin',
 2: 'Forehead - pull hairline',
 0: 'Glasses on/off',
 3: 'Neck - scratch',
 4: 'Neck - pinch skin',
 5: 'Eyelash - pull hair',
 6: 'Eyebrow - pull hair',
 7: 'Forehead - scratch',
 8: 'Above ear - pull hair'}


def add_mean_and_std_tof_thm(df):


    df["flight_mean"] = df.filter(like="tof_").mean(axis=1)
    df["flight_std"] = df.filter(like="tof_").std(axis=1)

    df["heat_mean"] = df.filter(like="thm_").mean(axis=1)
    df["heat_std"] = df.filter(like="thm_").std(axis=1)

    return df


def average_intervals(df):

    n = len(df)
    interval_size = n // 4
    intervals = [df.iloc[i*interval_size:(i+1)*interval_size] for i in range(4)]
    
    # Handle remainder by adding leftover rows to the last interval
    if n % 4 != 0:
        intervals[-1] = pd.concat([intervals[-1], df.iloc[4*interval_size:]])
    
    summary_rows = []
    for idx, interval in enumerate(intervals):
        avg_row = interval.mean(numeric_only=True)
        avg_row["interval"] = idx + 1
    
        # For categorical columns, take the first value in the interval
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for cat_col in categorical_cols:
            avg_row[cat_col] = interval.iloc[0][cat_col]
    
        summary_rows.append(avg_row)
    
    # Convert list of Series into one DataFrame, stacked vertically
    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    print(summary_df.shape)
    print(summary_df.columns)
    return summary_df

def flatten(df):
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove 'sequence_id' and 'interval' from numerical columns if present
    numerical_cols = [col for col in numerical_cols if col not in ["sequence_id", "interval"]]

    num_values = df[numerical_cols].values.flatten()
    # Take the first value for each categorical column
    cat_values = df[categorical_cols].iloc[0].values
    # Concatenate numerical and categorical values
    concat = np.concatenate([num_values, cat_values])

    # Build column names: numerical columns for each interval, then categorical columns
    num_col_names = []
    for i in range(len(df)):
        num_col_names.extend([f"{col}_interval{i+1}" for col in numerical_cols])
    col_names = num_col_names + categorical_cols

    return pd.DataFrame([concat], columns=col_names)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:

    with open("/kaggle/input/tof_and_thm_removed/other/default/1/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Convert Polars -> Pandas
    sequence_pd = sequence.to_pandas()

    sequence_pd = remove_thm_outliers(sequence_pd)
    sequence_pd = impute_tof_values(sequence_pd)
    sequenc_pd = add_mean_and_std_tof_thm(sequence_pd)
    #sequence_pd = average_tof_values(sequence_pd)
    demographics_pd = demographics.to_pandas()
    
    columns_to_drop = ["row_id", "sequence_counter"]
    sequence_pd = drop_columns(columns_to_drop, sequence_pd)

    # Merge demographics
    sequence_pd = average_intervals(sequence_pd)
    sequence_pd = flatten(sequence_pd)
    
    #sequence_pd = sequence_pd.merge(demographics_pd, on="subject", how="left")
    sequence_pd = sequence_pd.drop(["subject", "sequence_id"], axis=1)

    sequence_pd = sequence_pd.drop(
    columns=[
        col for col in sequence_pd.columns 
        if col == "gesture" or "tof" in col.lower() or "thm" in col.lower() or "heat" in col.lower() or "flight" in col.lower()
    ])
    
    # Predict gesture
    y_pred = model.predict(sequence_pd)
    
    gesture = inv_gesture_numerical_dict[int(y_pred[0])]
    print(y_pred)
    
    return gesture




# Serve the inference function
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




