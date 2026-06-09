pip install polars


import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from collections import Counter
import os

import kaggle_evaluation.cmi_inference_server


train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")

train_df = train_df.loc[train_df['sequence_type'] == 'Target'].reset_index(drop = True)

sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']

all_sensor_cols = [col for col in train_df.columns if any(s in col for s in ['acc_', 'rot_', 'thm_', 'tof_'])]

train_df[all_sensor_cols] = train_df[all_sensor_cols].fillna(-1)


le = LabelEncoder()
train_df['encoded_gesture'] = le.fit_transform(train_df['gesture'])


for i in train_df.columns:
    print(i)


def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    # Binary F1: assuming 'Target' is 1, 'Non-Target' is 0
    # Since we only trained on 'Target', our model will always predict a BFRB gesture.
    # Therefore, y_pred_binary will effectively always be 1 for a model trained this way.
    # This metric part might be misleading on a validation set that only contains 'Target' gestures.
    # For a proper binary F1, the true labels would need to include non-BFRB types.
    y_true_binary = np.ones_like(y_true, dtype=int) # All are 'Target' in this filtered dataset
    y_pred_binary = np.ones_like(y_pred, dtype=int) # Model predicts only BFRB if trained on 'Target'
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)

    # Macro F1: specific gesture classification
    # This is calculated only over the BFRB gestures.
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    final_score = (binary_f1 + macro_f1) / 2
    return final_score

unique_sequences = train_df[['sequence_id', 'encoded_gesture']].drop_duplicates()
unique_sequence_ids = unique_sequences['sequence_id']
unique_sequence_targets = unique_sequences['encoded_gesture']



splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(unique_sequence_ids, unique_sequence_targets):
    train_seq_ids = unique_sequence_ids.iloc[train_idx]
    val_seq_ids = unique_sequence_ids.iloc[val_idx]

X_train_df = train_df[train_df['sequence_id'].isin(train_seq_ids)]
X_val_df = train_df[train_df['sequence_id'].isin(val_seq_ids)]

y_train = X_train_df['encoded_gesture']
y_val = X_val_df['encoded_gesture']

X_train = X_train_df[sensor_cols]
X_val = X_val_df[sensor_cols]


rf_model = RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)

all_original_gestures_in_train = train_df['gesture'].unique()
validation_score = competition_metric(y_val, y_val_pred, le, all_original_gestures_in_train)
print(f"Validation Score: {validation_score:.4f}")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence_pd = sequence.to_pandas()
    sequence_pd[all_sensor_cols] = sequence_pd[all_sensor_cols].fillna(-1)
    X_inference = sequence_pd[sensor_cols]
    
    # Get predictions for all rows in the sequence
    all_predicted_label_ids = rf_model.predict(X_inference)
    
    # Find the most common predicted label ID
    # Counter returns a list of (element, count) tuples, sorted by count.
    # We want the element of the first tuple.
    most_common_label_id = Counter(all_predicted_label_ids).most_common(1)[0][0]
    
    # Convert the most common numerical label back to the original gesture string
    predicted_gesture_str = le.inverse_transform([most_common_label_id])[0]
    
    return predicted_gesture_str


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




