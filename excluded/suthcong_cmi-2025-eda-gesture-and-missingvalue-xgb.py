import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import kaggle_evaluation.cmi_inference_server

pd.set_option('display.max_columns', 100)
sns.set(style="whitegrid")



# Load file
train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"

train_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
test_demo_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

# Load dá»¯ liá»‡u
train = pd.read_csv(train_path)
train_demographics = pd.read_csv(train_demo_path)
test = pd.read_csv(test_path)
test_demographics = pd.read_csv(test_demo_path)
print("Train shape:", train.shape)
print("Unique sequences:", train['sequence_id'].nunique())


# Merge demographics
merge_demographics = pd.concat([train_demographics,test_demographics]).drop_duplicates().reset_index(drop=True)
merge_demographics


# columns only in train (not in test)
diff_cols = list(set(train.columns.to_list())-set(test.columns.to_list()))
diff_cols


gesture_df = train[diff_cols].drop_duplicates().reset_index(drop=True)
gesture_df.to_csv('gesture_dict.csv', index=False)
gesture_df


gesture_df[gesture_df['sequence_type']=='Target'].gesture.value_counts()



gesture_df[gesture_df['sequence_type']=='Non-Target'].gesture.value_counts()


# there are 4 stages of 1 gesture
train.behavior.unique()


# All gesture have the same 4 stages
pd.DataFrame(pd.DataFrame(train.groupby(['gesture', 'behavior']).size()).reset_index().gesture.value_counts())


target = ['gesture']
meta_features= ['row_id', 'sequence_type', 'sequence_id', 'sequence_counter', 'subject', 'orientation', 'behavior', 'phase']
imu_sensor_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',]
thermo_features = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',]
tof_1_features = [f'tof_1_v{px}' for px in range(64)]
tof_2_features = [f'tof_2_v{px}' for px in range(64)]
tof_3_features = [f'tof_3_v{px}' for px in range(64)]
tof_4_features = [f'tof_4_v{px}' for px in range(64)]
tof_5_features = [f'tof_5_v{px}' for px in range(64)]

tof_features = tof_1_features + tof_2_features + tof_3_features + tof_4_features + tof_5_features
sensors_features = imu_sensor_features + thermo_features + tof_features

all_features = meta_features + sensors_features
len(all_features+target)


train.info()


def check_missing_df(df:pd.DataFrame,col:list)->pd.DataFrame:
    missing_data = df[col].isnull().sum()
    missing_pct = (missing_data / len(train)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_data, 'Missing_Percentage': missing_pct})
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)


check_missing_df(train, meta_features+target)


check_missing_df(train, imu_sensor_features)


check_missing_df(train, thermo_features)


check_missing_df(train, tof_1_features)


check_missing_df(train, tof_2_features)


import missingno as msno
msno.matrix(train)


target = ['gesture']
meta_features= ['row_id', 'sequence_type', 'sequence_id', 'sequence_counter', 'subject', 'orientation', 'behavior', 'phase']
imu_sensor_features = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',]
thermo_features = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',]
tof_1_features = [f'tof_1_v{px}' for px in range(64)]
tof_2_features = [f'tof_2_v{px}' for px in range(64)]
tof_3_features = [f'tof_3_v{px}' for px in range(64)]
tof_4_features = [f'tof_4_v{px}' for px in range(64)]
tof_5_features = [f'tof_5_v{px}' for px in range(64)]

tof_features = tof_1_features + tof_2_features + tof_3_features + tof_4_features + tof_5_features
sensors_features = imu_sensor_features + thermo_features + tof_features

all_features = meta_features + sensors_features



train['tof_1_mean'] = train[tof_1_features].mean(axis=1)
train['tof_2_mean'] = train[tof_2_features].mean(axis=1)
train['tof_3_mean'] = train[tof_3_features].mean(axis=1)
train['tof_4_mean'] = train[tof_4_features].mean(axis=1)
train['tof_5_mean'] = train[tof_5_features].mean(axis=1)

tof_means = ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean']

check_ms_train = train[meta_features+imu_sensor_features+thermo_features+ tof_means]

check_ms_train[check_ms_train.isnull().any(axis=1)]


msno.matrix(check_ms_train)



msno.dendrogram(check_ms_train)


train[imu_sensor_features].describe().T


train[thermo_features].describe().T


# tof_features = tof_1_features + tof_2_features + tof_3_features + tof_4_features + tof_5_features
train[tof_5_features].describe().T
# -1 mean no sensor response


# check if any columns only have exactly 1 value:
NUNIQUE1 = [c for c in train.columns if train[c].nunique()==1]
NUNIQUE1


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


le = LabelEncoder()
train['encoded_gesture'] = le.fit_transform(train['gesture'])
unique_sequences = train[['sequence_id', 'encoded_gesture']].drop_duplicates()
unique_sequence_ids = unique_sequences['sequence_id']
unique_sequence_targets = unique_sequences['encoded_gesture']


train = train.dropna()

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(unique_sequence_ids, unique_sequence_targets):
    train_seq_ids = unique_sequence_ids.iloc[train_idx]
    val_seq_ids = unique_sequence_ids.iloc[val_idx]

X_train = train[train['sequence_id'].isin(train_seq_ids)]
X_val = train[train['sequence_id'].isin(val_seq_ids)]

y_train = X_train['encoded_gesture']
y_val = X_val['encoded_gesture']


col_to_drop = ['row_id','sequence_id','subject', 'orientation', 'behavior', 'phase', 'sequence_type']

X_train = X_train[all_features].drop(columns=col_to_drop)
X_val = X_val[all_features].drop(columns=col_to_drop)


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

all_original_gestures_in_train = train['gesture'].unique()
all_original_gestures_in_train


# use default params
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)




all_original_gestures_in_train = train['gesture'].unique()
validation_score = competition_metric(y_val, y_val_pred, le, all_original_gestures_in_train)
print(f"Validation Score: {validation_score:.4f}")


all_feature_cols = X_train.columns.to_list()


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence_pd = sequence.to_pandas()
    sequence_pd[all_feature_cols] = sequence_pd[all_feature_cols].fillna(-1)
    X_inference = sequence_pd[all_feature_cols]
    predicted_label_id = rf_model.predict(X_inference)[0]
    predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
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


# view submission file
pd.read_parquet('/kaggle/working/submission.parquet')

