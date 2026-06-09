# === Core Python & Utilities ===
import os
import joblib
from concurrent.futures import ProcessPoolExecutor

# === Numerical & Data Handling ===
import numpy as np
import pandas as pd
import dask.dataframe as dd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

# === Plotting ===
import matplotlib.pyplot as plt

# === Machine Learning ===
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# === TensorFlow / Keras ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import (
    Layer, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    Activation, Dense, Flatten, Reshape
)



def custom_loss_improved(y_true, y_pred):
    # Weighted focal MSE
    alpha = 6.0
    gamma = 2.0
    weight = 1 + alpha * K.abs(y_true)
    error = y_true - y_pred
    focal_weight = K.pow(K.abs(error), gamma)
    mse_loss = K.mean(weight * focal_weight * K.square(error))

    # Your original ranking loss (unchanged)
    pos_mask = K.cast(K.greater(K.abs(y_true), 0.6), 'float32')
    neg_mask = 1.0 - pos_mask
    pos_count = K.sum(pos_mask) + K.epsilon()
    neg_count = K.sum(neg_mask) + K.epsilon()
    pos_mean = K.sum(K.abs(y_pred) * pos_mask) / pos_count
    neg_mean = K.sum(K.abs(y_pred) * neg_mask) / neg_count
    margin = 0.6
    ranking_loss = K.relu(margin - (pos_mean - neg_mean))

    return mse_loss + 0.1 * ranking_loss


# Load the worn-unworn model
xgboost_worn_unworn_model = XGBClassifier()
xgboost_worn_unworn_model.load_model('/kaggle/input/worn-unworn-latest/worn_unworn_xgboost_model_2.json')
print("ðŸš€ Worn-Unworn predict XGboost model loaded successfully!")

# load score prediction models (unet, lstm)

unet_model_path = "/kaggle/input/exp-score/unet1d_with_attention_exp_score_loss_2_earlystop.keras"

score_predict_unet_model = keras.models.load_model(
    unet_model_path,
    custom_objects={
        "custom_loss_improved": custom_loss_improved
    }
)

print("ðŸš€ Score predict UNET model loaded successfully!")

lstm_model_path = "/kaggle/input/exp-score/lstm_with_attention_exp_score_loss_2_earlystop.keras"

score_predict_lstm_model = keras.models.load_model(
    lstm_model_path,
    custom_objects={
        "custom_loss_improved": custom_loss_improved
            }
)
print("ðŸš€ Score predict LSTM model loaded successfully!")



# Load test data

test_series = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet')

# Define a custom bucket: 18:00 to 18:00 the following day
def assign_bucket(timestamp):
    if timestamp.hour >= 18:
        return timestamp.date()
    else:
        return (timestamp - pd.Timedelta(days=1)).date()

def process_series_df(series_id):
    # Step 1: Check if required columns exist
    required_cols = ['series_id', 'timestamp', 'anglez', 'enmo', 'step']
    missing_cols = [col for col in required_cols if col not in test_series.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return []

    # Step 2: Filter the series
    series_df = test_series[test_series['series_id'] == series_id].copy()
    if series_df.empty:
        return []

    # Step 3: Handle missing anglez column
    if series_df['anglez'].isna().all():
        return []

    # Step 4: Fill missing values in relevant columns (forward fill)
    series_df[['timestamp', 'anglez', 'enmo', 'step']] = (
        series_df[['timestamp', 'anglez', 'enmo', 'step']].ffill()
    )

    # Step 5: Convert timestamp safely
    series_df['timestamp'] = pd.to_datetime(series_df['timestamp'], errors='coerce', utc=True)
    if series_df['timestamp'].isna().all():
        return []

    # Step 6: Extract time-based features
    series_df['date_hour_minute'] = series_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    series_df['hour'] = series_df['timestamp'].dt.hour

    # Step 7: Compute anglez value counts
    series_df['anglez_value_counts'] = series_df['anglez'].map(series_df['anglez'].value_counts())

    # Step 8: Group by minute
    series_minute_grouped = series_df.groupby('date_hour_minute').agg({
        'anglez': ['mean', 'std', 'median'],
        'enmo': ['mean', 'std', 'median'],
        'step': 'min',
        'hour': 'first',
        'anglez_value_counts': 'max',
    }).reset_index()

    if series_minute_grouped.empty:
        return []

    # Step 9: Rename columns
    series_minute_grouped.columns = [
        'date_hour_minute', 'anglez_mean', 'anglez_std', 'anglez_med',
        'enmo_mean', 'enmo_std', 'enmo_med', 'step_min',
        'hour', 'anglez_value_counts'
    ]

    # Step 10: Parse date_hour_minute and assign bucket
    series_minute_grouped['date_hour_minute'] = pd.to_datetime(
        series_minute_grouped['date_hour_minute'], errors='coerce'
    )

    series_minute_grouped['bucket'] = series_minute_grouped['date_hour_minute'].apply(assign_bucket)

    return series_minute_grouped
      
def process(series_id):
  print(f"start processing {series_id}")
  series_minute_grouped = process_series_df(series_id)
  if series_minute_grouped.empty:
        print(f"No valid data for series_id {series_id}")
        return []
  series_minute_grouped['bucket'] = series_minute_grouped['bucket'].astype(str)

  # Group data into buckets (dates)
  buckets = series_minute_grouped.groupby('bucket')

  # Store data for each bucket as arrays
  bucket_arrays = []

  for bucket, group in buckets:
      step = group['step_min'].to_numpy()
      hour = group['hour'].to_numpy()
      enmo_values = group['enmo_mean'].to_numpy()
      enmo_std_values = group['enmo_std'].to_numpy()
      anglez_values = group['anglez_mean'].to_numpy()
      anglez_values = np.clip(anglez_values, -90, 90)
      anglez_log_values = np.log1p(anglez_values + 91)  # Apply shift by 91 so the range is 1-181 and log1p transformation
      anglez_std_values = group['anglez_std'].to_numpy()
      anglez_value_counts = group['anglez_value_counts'].to_numpy()
      bucket_arrays.append({
          'series_id': series_id,
          'bucket': bucket,
          'step': step,
          'hour': hour,
          'enmo': enmo_values,
          'enmo_std' : enmo_std_values,
          'anglez_log': anglez_log_values,
          'anglez_std' : anglez_std_values,
          'anglez_value_counts' : anglez_value_counts,
      })
  print(f"finished processing {series_id}")
  return bucket_arrays


TARGET_LENGTH = 1440

def pad_to_length_edge(array, target_length=1440):
    pad_length = target_length - len(array)
    if pad_length > 0:
        return np.pad(array, (0, pad_length), mode='edge')
    return array

series_ids = test_series['series_id'].unique()
results = [process(series_id) for series_id in series_ids]

flat_data = []
for result in results:
    for bucket in result:
        flat_data.append({
            "series_id": bucket['series_id'],
            "bucket": bucket['bucket'],
            "step": bucket['step'],
            "hour": bucket['hour'],
            "enmo": list(bucket['enmo']),
            "enmo_std": list(bucket['enmo_std']),
            "anglez_log": list(bucket['anglez_log']),
            "anglez_std": list(bucket['anglez_std']),
            "anglez_value_counts": list(bucket['anglez_value_counts']),
        })

buckets = pd.DataFrame(flat_data)

# Apply padding to each feature column
for col in ['step', 'enmo', 'hour', 'enmo_std', 'anglez_log', 'anglez_std', 'anglez_value_counts']:
    buckets[col] = buckets[col].apply(lambda x: pad_to_length_edge(x, 1440))


buckets['anglez_log_ewma'] = buckets['anglez_log'].apply(lambda x: pd.Series(x).ewm(span=30).mean().to_numpy())
buckets['enmo_ewma'] = buckets['enmo'].apply(lambda x: pd.Series(x).ewm(span=30).mean().to_numpy())
for col_name in ['enmo', 'enmo_std', 'anglez_log', 'anglez_std', 'anglez_log_ewma', 'enmo_ewma']:
  buckets[col_name] = buckets[col_name].apply(lambda arr: (arr - np.mean(arr)) / np.std(arr) if np.std(arr) != 0 else arr)


# Efficient sliding window std
def sliding_window_std(array, window_size=10):
    if len(array) < window_size:
        return np.array([0] * len(array))  # Handle edge cases
    windows = sliding_window_view(array, window_shape=window_size)  # Create overlapping windows
    return np.std(windows, axis=1)

# Efficient sliding window mad
def sliding_window_mad(array, window_size=10):
    if len(array) < window_size:
        return np.array([0] * len(array))  # Handle edge cases
    windows = sliding_window_view(array, window_shape=window_size)  # Create overlapping windows
    return np.mean(np.abs(np.diff(windows, axis=1)), axis=1)
def pad_to_length_constant(array, target_length=1440):
    pad_length = target_length - len(array)
    if pad_length > 0:
        return np.pad(array, (0, pad_length), mode='constant')
    return array
buckets['anglez_log_std'] = buckets['anglez_log'].apply(lambda x: sliding_window_std(np.array(x), window_size=30))
buckets['anglez_log_mad'] = buckets['anglez_log'].apply(lambda x: sliding_window_mad(np.array(x), window_size=30))


# Pad the sliding window results
buckets['anglez_log_std'] = buckets['anglez_log_std'].apply(lambda x: pad_to_length_edge(x, 1440))
buckets['anglez_log_mad'] = buckets['anglez_log_mad'].apply(lambda x: pad_to_length_edge(x, 1440))
buckets['hour_sin'] = buckets['hour'].apply(lambda x: np.sin(2 * np.pi * x / 24))
buckets['hour_cos'] = buckets['hour'].apply(lambda x: np.cos(2 * np.pi * x / 24))


# Feature engineering: Prepare features and targets
features = []
for _, row in buckets.iterrows():
    # Stack all features into a single array for the row
    feature_matrix = np.column_stack([
        row['anglez_log_ewma'],
        row['anglez_log_std'],
        row['anglez_log_mad'],
        row['enmo_ewma'],
        row['enmo_std'],
        row['anglez_std'],
        row['anglez_value_counts'],
        row['hour_sin'],
        row['hour_cos']
    ])
    features.append(feature_matrix)
# Convert features and targets to numpy arrays
X_test = np.array(features)  # Shape: (n_samples, 1440, n_features)

# Flatten the input arrays into 1D vectors
if X_test.size != 0:
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)  # Shape: (num_samples, features)
    print(f"Shape of flattened X: {X_test_flattened.shape}")
    worn_predictions = xgboost_worn_unworn_model.predict(X_test_flattened)
    buckets['worn'] = worn_predictions
    filtered_buckets = buckets[buckets['worn'] == 1]
else:
    filtered_buckets = buckets


# Extract features
features = []
for _, row in filtered_buckets.iterrows():
# Stack features: each is a 1440-length array for one day
    feature_matrix = np.column_stack([
    row['anglez_log_ewma'],
    row['anglez_log_std'],
    row['anglez_log_mad'],
    row['enmo_ewma'],
    row['enmo_std'],
    row['anglez_std'],
    row['anglez_value_counts'],
    row['hour_sin'],
    row['hour_cos']
    ])
    features.append(feature_matrix)

X = np.array(features)  # Ensure shape: (n_samples, 1440, n_features)

# Predict using the trained models
sleep_awake_predictions_1 = score_predict_unet_model.predict(X)
sleep_awake_predictions_2 = score_predict_lstm_model.predict(X)

# Assign predictions to the DataFrame  
filtered_buckets["predicted_target"] = [(0.6*sleep_awake_predictions_1[i].flatten()+0.4*sleep_awake_predictions_2[i].flatten()) for i in range(len(filtered_buckets))]

# Save results
filtered_buckets.to_parquet("buckets_for_post_processing.parquet", index=False)



# Load data using Dask
df = dd.read_parquet('buckets_for_post_processing.parquet')

threshold_onset, threshold_wakeup = 0.01, 0.01

# Step adjustment based on empirical insights
def adjust_step(step, event_type):
    step = int(step)  # ensure it's a standard Python integer
    if step % 12 in [0, 6]:
        step -= 1
    e15 = event_type[0] + str((step // 12) % 15)
    if e15 in ['o1', 'o8', 'o9', 'w0', 'w4', 'w8', 'w12']:
        step -= 12
    elif e15 in ['o4', 'o12', 'w1', 'w5', 'w9', 'w13']:
        step += 12
    return step

# Comprehensive event extraction accounting for sleep breaks
def process_partition(partition_df):
    min_sleep_duration, max_sleep_duration = 60 * 12, 1000 * 12
    max_sleep_break = 30 * 12  # 30 minutes in steps
    submission_rows = []

    for idx, row in partition_df.iterrows():
        predicted_scores = row['predicted_target']
        if predicted_scores is None:
          print(idx)
          continue  # Skip rows where predicted_scores are None

        predicted_scores = np.array(predicted_scores).flatten()

        steps = np.array(row['step'], dtype=np.int64) 

        onset_peaks, _ = find_peaks(predicted_scores, height=threshold_onset)
        wakeup_peaks, _ = find_peaks(-predicted_scores, height=threshold_wakeup)

        onset_steps_adjusted = [adjust_step(steps[idx], 'onset') for idx in onset_peaks]
        wakeup_steps_adjusted = [adjust_step(steps[idx], 'wakeup') for idx in wakeup_peaks]

        onset_scores = predicted_scores[onset_peaks]
        wakeup_scores = predicted_scores[wakeup_peaks]

        for onset_step, onset_score in zip(onset_steps_adjusted, onset_scores):
            for wakeup_step, wakeup_score in zip(wakeup_steps_adjusted, wakeup_scores):
                duration = wakeup_step - onset_step
                if duration < 0:
                    continue
                if min_sleep_duration <= duration <= max_sleep_duration or (0 < duration <= max_sleep_break):
                  if abs(onset_score) > threshold_onset:
                      submission_rows.append({
                          'series_id': str(row['series_id']),
                          'step': int(onset_step),
                          'event': 'onset',
                          'score': float(abs(onset_score))
                      })

                  if abs(wakeup_score) > threshold_wakeup:
                      submission_rows.append({
                          'series_id': str(row['series_id']),
                          'step': int(wakeup_step),
                          'event': 'wakeup',
                          'score': float(abs(wakeup_score))
                      })
    if submission_rows:
        return pd.DataFrame(submission_rows).drop_duplicates()
    else:
        return pd.DataFrame(columns=['series_id', 'step', 'event', 'score'])

# Apply processing to each partition
results = df.map_partitions(process_partition, meta={
    'series_id': str,
    'step': int,
    'event': str,
    'score': float
})

# Compute results, assign row_id, and save
submission_df = results.reset_index(drop=True).compute()
submission_df['row_id'] = submission_df.index
submission_df = submission_df[['row_id', 'series_id', 'step', 'event', 'score']]
submission_df.to_csv('submission.csv', index=False)

