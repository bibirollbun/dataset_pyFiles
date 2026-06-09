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


train_data = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_data = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")


test_demo.head()


train_data.sample(n=5,random_state=42)


train_data.shape


test_data.head()


train_demo.head()


train_demo.shape


# Merge the train data and demogrphics data 
train_data = train_data.merge(train_demo, on='subject', how='left')
test_data = test_data.merge(test_demo, on='subject', how='left')


train_data.head()


# Handiling missing data
missing_count = test_data.isnull().sum()
missing_count = missing_count[missing_count>0]
print('missing_counts: ',missing_count)


train_data.isnull().sum()[train_data.isnull().sum() > 0]


train_data.describe()


# Replace NaN values with mean of their respective columns
train_data.fillna(train_data.mean(numeric_only=True), inplace=True)


train_data.describe()


# Use Label Encoder to encode
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_data['orientation'] = le.fit_transform(train_data['orientation'])


# Feature Selection
from sklearn.preprocessing import StandardScaler

sensor_cols = [col for col in train_data.columns if col.startswith(('acc_', 'gyr_', 'therm_', 'tof_'))]
demo_cols = ['age','height_cm','shoulder_to_wrist_cm','elbow_to_wrist_cm']

scaler=StandardScaler()
train_data[sensor_cols] = train_data.groupby('sequence_id')[sensor_cols].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
train_data[demo_cols] = scaler.fit_transform(train_data[demo_cols])


# Define IMU cols and demo cols
imu_cols = [col for col in train_data.columns if col.startswith(('acc_','gyr_'))]
demo_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'] + \
            [col for col in train_data.columns if col.endswith('_encoded') or col.startswith(('sex_', 'handedness_', 'adult_child_', 'orientation_'))]


# Aggregate time-series features for gesture phase
gesture_df = train_data[train_data['phase'] == 'Gesture']
print(f"Gesture DataFrame shape: {gesture_df.shape}")
assert not gesture_df.empty, "Error: gesture_df is empty. Check if 'phase' contains 'Gesture'."
agg_funcs = ['mean', 'std', 'min', 'max', 'median']
gesture_features = gesture_df.groupby('sequence_id')[sensor_cols].agg(agg_funcs).reset_index()
gesture_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in gesture_features.columns]
print(f"Gesture features shape: {gesture_features.shape}")


import warnings
warnings.filterwarnings('ignore')

# Compute Time-Series Difference Features
diff_df = gesture_df[['sequence_id']].copy()
for col in sensor_cols:
    diff_df[f'{col}_diff'] = gesture_df.groupby('sequence_id')[col].diff().fillna(0)
gesture_df = pd.concat([gesture_df, diff_df[[col for col in diff_df.columns if col.endswith('_diff')]]], axis=1)
diff_cols = [f'{col}_diff' for col in sensor_cols]
diff_features = gesture_df.groupby('sequence_id')[diff_cols].agg(agg_funcs).reset_index()
diff_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in diff_features.columns]
features = gesture_features.merge(diff_features, on='sequence_id', how='left')
print(f"Features after diff merge shape: {features.shape}")


# Incorporate Demographic Features
demo_features = train_data.groupby('sequence_id')[demo_cols].first().reset_index()
features = features.merge(demo_features, on='sequence_id', how='left')
print(f"Features after demo merge shape: {features.shape}")


# Add Contextual Features from Transition and Pause Phases
transition_df = train_data[train_data['phase'] == 'Transition']
pause_df = train_data[train_data['phase'] == 'Pause']
print(f"Transition DataFrame shape: {transition_df.shape}")
print(f"Pause DataFrame shape: {pause_df.shape}")
transition_features = transition_df.groupby('sequence_id')[sensor_cols].agg(agg_funcs).reset_index()
transition_features.columns = ['transition_' + '_'.join(col).strip() if col[1] else col[0] for col in transition_features.columns]
pause_features = pause_df.groupby('sequence_id')[sensor_cols].agg(agg_funcs).reset_index()
pause_features.columns = ['pause_' + '_'.join(col).strip() if col[1] else col[0] for col in pause_features.columns]
features = features.merge(transition_features, on='sequence_id', how='left')
features = features.merge(pause_features, on='sequence_id', how='left')
features = features.fillna(0)
print(f"Features after transition/pause merge shape: {features.shape}")


# Create IMU-Only and Full-Sensor Feature Sets
imu_features = features[[col for col in features.columns if any(col.startswith(c) for c in imu_cols) or
                        col in demo_cols or
                        col.startswith(('transition_acc_', 'transition_gyr_', 'pause_acc_', 'pause_gyr_')) or
                        col == 'sequence_id']]
full_features = features
print(f"IMU features shape: {imu_features.shape}")
print(f"Full features shape: {full_features.shape}")

# 2.6: Add Labels
labels = train_data.groupby('sequence_id').agg({
    'sequence_type': 'first',
    'gesture': 'first'
}).reset_index()
labels['is_bfrb'] = (labels['sequence_type'] == 'target').astype(int)
missing_seqs = set(full_features['sequence_id']).difference(labels['sequence_id'])
if missing_seqs:
    print(f"Warning: {len(missing_seqs)} sequence_ids in features not found in labels: {missing_seqs}")
imu_features = imu_features.merge(labels[['sequence_id', 'is_bfrb', 'gesture']], on='sequence_id', how='left')
full_features = full_features.merge(labels[['sequence_id', 'is_bfrb', 'gesture']], on='sequence_id', how='left')
print(f"IMU features after labels merge shape: {imu_features.shape}")
print(f"Full features after labels merge shape: {full_features.shape}")


# 2.7: Feature Selection for Random Forest
from sklearn.ensemble import RandomForestClassifier

X_full = full_features.drop(['sequence_id', 'gesture', 'is_bfrb', 'sequence_type'], axis=1, errors='ignore')
y_full = full_features['gesture']
print(f"X_full shape for feature selection: {X_full.shape}")
assert X_full.shape[0] > 0, f"Error: X_full has zero samples (shape: {X_full.shape}). Check previous steps for data loss."
if X_full.shape[0] > 1:  # Only perform feature selection if enough samples
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_full, y_full)
    feature_importance = pd.Series(rf_model.feature_importances_, index=X_full.columns)
    top_features = feature_importance.nlargest(50).index
    full_features_selected = full_features[['sequence_id', 'gesture', 'is_bfrb'] + list(top_features)]
    imu_top_features = [col for col in top_features if col in imu_features.columns]
    imu_features_selected = imu_features[['sequence_id', 'gesture', 'is_bfrb'] + imu_top_features]
else:
    print("Warning: Insufficient samples for feature selection. Using all features.")
    full_features_selected = full_features
    imu_features_selected = imu_features
print(f"Selected full features shape: {full_features_selected.shape}")
print(f"Selected IMU features shape: {imu_features_selected.shape}")


# Split Data for Random Forest
from sklearn.model_selection import train_test_split

train_seq, val_seq = train_test_split(
    full_features_selected['sequence_id'],
    test_size=0.2,
    stratify=full_features_selected['gesture'],
    random_state=42
)
train_full = full_features_selected[full_features_selected['sequence_id'].isin(train_seq)]
val_full = full_features_selected[full_features_selected['sequence_id'].isin(val_seq)]
train_imu = imu_features_selected[imu_features_selected['sequence_id'].isin(train_seq)]
val_imu = imu_features_selected[imu_features_selected['sequence_id'].isin(val_seq)]
print(f"Train full shape: {train_full.shape}")
print(f"Val full shape: {val_full.shape}")
print(f"Train IMU shape: {train_imu.shape}")
print(f"Val IMU shape: {val_imu.shape}")


# Full-Sensor Random Forest
X_train_full = train_full.drop(['sequence_id', 'gesture', 'is_bfrb'], axis=1)
y_train_full = train_full['gesture']
X_val_full = val_full.drop(['sequence_id', 'gesture', 'is_bfrb'], axis=1)
y_val_full = val_full['gesture']
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_full, y_train_full)
y_pred_full = rf_full.predict(X_val_full)


from sklearn.metrics import f1_score
import joblib

# Custom evaluation metric (fixed)
def custom_f1_score(y_true, y_pred):
    # Binary F1: Map to 1 (BFRB-like) or 0 (non_target)
    binary_true = [1 if g != 'non_target' else 0 for g in y_true]
    binary_pred = [1 if g != 'non_target' else 0 for g in y_pred]
    try:
        binary_f1 = f1_score(binary_true, binary_pred, average='binary')
    except ValueError as e:
        print(f"Error computing Binary F1: {e}. Returning 0.")
        binary_f1 = 0.0
    # Macro F1: Multiclass F1
    try:
        macro_f1 = f1_score(y_true, y_pred, average='macro')
    except ValueError as e:
        print(f"Error computing Macro F1: {e}. Returning 0.")
        macro_f1 = 0.0
    return (binary_f1 + macro_f1) / 2

score_full = custom_f1_score(y_val_full, y_pred_full)
print(f"Random Forest Custom F1 Score (Full Sensors): {score_full}")
joblib.dump(rf_full, 'rf_full_model.pkl')

# IMU-Only Random Forest
X_train_imu = train_imu.drop(['sequence_id', 'gesture', 'is_bfrb'], axis=1)
y_train_imu = train_imu['gesture']
X_val_imu = val_imu.drop(['sequence_id', 'gesture', 'is_bfrb'], axis=1)
y_val_imu = val_imu['gesture']
rf_imu = RandomForestClassifier(n_estimators=100, random_state=42)
rf_imu.fit(X_train_imu, y_train_imu)
y_pred_imu = rf_imu.predict(X_val_imu)
score_imu = custom_f1_score(y_val_imu, y_pred_imu)
print(f"Random Forest Custom F1 Score (IMU-Only): {score_imu}")
joblib.dump(rf_imu, 'rf_imu_model.pkl')




