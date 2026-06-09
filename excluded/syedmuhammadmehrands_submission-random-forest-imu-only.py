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


import os
import pandas as pd
import polars as pl
import joblib



feature_names = ['acc_x_mean',	'acc_x_std',	'acc_y_mean',	'acc_y_std',	'acc_z_mean',	'acc_z_std',	'rot_w_mean',	'rot_w_std',	'rot_x_mean',	'rot_x_std',	'rot_y_mean',	'rot_y_std',	'rot_z_mean',	'rot_z_std',	'behavior_freq_Hand at target location',	'behavior_freq_Moves hand to target location',	'behavior_freq_Performs gesture',	'behavior_freq_Relaxes and moves hand to target location',	'phase_freq_Gesture',	'phase_freq_Transition']


model = joblib.load('/kaggle/input/gbc/scikitlearn/default/4/gbc.joblib')
label_encoder = joblib.load("/kaggle/input/gbc/scikitlearn/default/4/label_encoder.joblib")


# Your feature extraction function — must match what you used during training
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_dict = {
        'acc_x_mean': df['acc_x'].mean(),
        'acc_y_mean': df['acc_y'].mean(),
        'acc_z_mean': df['acc_z'].mean(),
        'rot_x_mean': df['rot_x'].mean(),
        'rot_y_mean': df['rot_y'].mean(),
        'rot_z_mean': df['rot_z'].mean(),
        'acc_x_std': df['acc_x'].std(),
        'acc_y_std': df['acc_y'].std(),
        'acc_z_std': df['acc_z'].std(),
        'rot_x_std': df['rot_x'].std(),
        'rot_y_std': df['rot_y'].std(),
        'rot_z_std': df['rot_z'].std(),
        'rot_w_mean': df['rot_w'].mean(),
        'rot_w_std': df['rot_w'].std(),
        # Add other features you used in training...
    }
    return pd.DataFrame([feature_dict])


def frequency_encode_categoricals(df, categorical_cols):
    freq_features = []
    for col in categorical_cols:
        counts = pd.crosstab(df['sequence_id'], df[col], normalize='index')
        counts.columns = [f'{col}_freq_{c}' for c in counts.columns]
        freq_features.append(counts)
    return pd.concat(freq_features, axis=1).reset_index()


# The main function called for each test sequence
# def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
#     df = sequence.to_pandas()
    
#     # Extract sequence_id for merging later
#     seq_id = df['sequence_id'].iloc[0]

#     # Statistical features
#     stat_features = extract_features(df)
#     stat_features['sequence_id'] = seq_id

#     # Frequency-encoded features
#     #freq_features = frequency_encode_categoricals(df, ['phase'])
    
#     # Merge statistical + frequency features
#     #features = pd.merge(stat_features, freq_features, on='sequence_id', how='left')
#     features = stat_features
#     features = features.drop(columns=['sequence_id'])  # Drop seq_id before predict
#     print(features.head())
#     # Fill missing columns if needed (due to unseen categories)
#     for col in model.feature_names_in_:
#         if col not in features.columns:
#             features[col] = 0.0
#     features = features[model.feature_names_in_]  # Ensure same column order
#     print('2nd Feature Head:\n', features.head())
#     # Predict

#     pred_int = model.predict(features)              # returns [7]
#     print("Prediction\n", pred_int)
#     pred_labels = label_encoder.inverse_transform(pred_int)  # returns ['n']    
#     print('print Labels:\n', pred_labels)
#     return pred_labels[0]




# def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
#     try:
#         df = sequence.to_pandas()
#         seq_id = df["sequence_id"].iloc[0]
        
#         # Check for missing IMU data
#         imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
#         imu_all_missing = df[imu_cols].isnull().all().all()

#         if imu_all_missing:
#             print(f"Sequence {seq_id}: IMU missing — using fallback label.")
#             return "Other"  # Use the most common class or a neutral fallback

#         features = extract_features(df)
#         # Replace with your model prediction code
#         pred = gbc.predict([features])[0]
#         label = label_encoder.inverse_transform([pred])[0]

#         print(f"Sequence {seq_id}: Predicted - {label}")
#         return label

#     except Exception as e:
#         print(f"Error in sequence {seq_id}: {e}")
#         return "Other"



def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        df = sequence.to_pandas()
        seq_id = df['sequence_id'].iloc[0]

        stat_features = extract_features(df)
        stat_features['sequence_id'] = seq_id

        features = stat_features.drop(columns=['sequence_id'])

        # Ensure all expected columns are present
        for col in feature_names:
            if col not in features.columns:
                features[col] = 0.0

        features = features[feature_names]
        features.fillna(0.0, inplace=True)

        pred = model.predict(features)[0]
        print(pred)
        label = label_encoder.inverse_transform([pred])[0]
        print(label)
        return label

    except Exception as e:
        print(f"Sequence {seq_id}: Exception during prediction - {e}")
        return "Other"



# Start the inference server
import kaggle_evaluation.cmi_inference_server
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




