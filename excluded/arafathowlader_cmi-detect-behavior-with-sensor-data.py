import numpy as np
import pandas as pd
import pickle
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import polars as pl
import os

import kaggle_evaluation.cmi_inference_server
import warnings
warnings.filterwarnings("ignore")


categorical_cols = ['sequence_type', 'orientation', 'phase', 'behavior']


train_data =  pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
numerical_cols = [col for col in train_data.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
train_data.drop(columns = categorical_cols, inplace=True)
train_data = train_data.groupby("sequence_id").apply(lambda x: x.fillna(method="ffill")).reset_index(drop=True)
train_data.fillna(method='bfill', inplace=True)

train_data["acc_magnitude"] = (train_data['acc_x']**2 + train_data['acc_y']**2 + train_data['acc_z']** 2)**0.5
train_data["rot_magnitude"] = (train_data['rot_x']**2 + train_data['rot_y']**2 + train_data['rot_z']** 2)**0.5
train_data['acc_x_diff'] = train_data.groupby('sequence_id')['acc_x'].diff().fillna(0)
train_data['acc_x_mean'] = train_data.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).mean().reset_index(0, drop= True)
train_data['acc_x_std'] = train_data.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).std().reset_index(0, drop= True)

for i in range(1, 6):
    tof_cols = [col for col in train_data.columns if col.startswith(f'tof_{i}_')]
    train_data[f'tof_{i}_mean'] = train_data[tof_cols].mean(axis = 1)
    train_data[f'tof_{i}_std'] = train_data[tof_cols].std(axis= 1)
    
for col in numerical_cols:
    train_data[col].fillna(train_data[col].mean(), inplace=True)

scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# subset = train_data[['sequence_id', 'sequence_counter', 'acc_x', 'acc_y', 'acc_z']]
# features = extract_features(
#     subset,
#     column_id='sequence_id',
#     column_sort='sequence_counter',
#     default_fc_parameters=MinimalFCParameters()
# )
# features = features.reset_index().rename(columns={'index': 'sequence_id'})
# train_data = train_data.merge(features, on='sequence_id', how='left')
train_data["acc_x_std"].fillna(train_data["acc_x_std"].mean(), inplace=True)


# train_data.to_parquet("train_processed.parquet")


## preprocess test data


test_data = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
numerical_cols = [col for col in test_data.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
test_data = test_data.groupby('sequence_id').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
test_data.fillna(method='bfill', inplace=True)
for col in numerical_cols:
    test_data[col].fillna(test_data[col].mean(), inplace=True)

test_data["acc_magnitude"] = (test_data['acc_x']**2 + test_data['acc_y']**2 + test_data['acc_z']** 2)**0.5
test_data["rot_magnitude"] = (test_data['rot_x']**2 + test_data['rot_y']**2 + test_data['rot_z']** 2)**0.5
test_data['acc_x_diff'] = test_data.groupby('sequence_id')['acc_x'].diff().fillna(0)
test_data['acc_x_mean'] = test_data.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).mean().reset_index(0, drop= True)
test_data['acc_x_std'] = test_data.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).std().reset_index(0, drop= True)

for i in range(1, 6):
    tof_cols = [col for col in test_data.columns if col.startswith(f'tof_{i}_')]
    test_data[f'tof_{i}_mean'] = test_data[tof_cols].mean(axis = 1)
    test_data[f'tof_{i}_std'] = test_data[tof_cols].std(axis= 1)

test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

subset_test = test_data[['sequence_id', 'sequence_counter', 'acc_x', 'acc_y', 'acc_z']]
features_test = extract_features(
    subset_test,
    column_id='sequence_id',
    column_sort='sequence_counter',
    default_fc_parameters=MinimalFCParameters()
)
features_test = features_test.reset_index().rename(columns={'index': 'sequence_id'})
test_data = test_data.merge(features_test, on='sequence_id', how='left')
test_data["acc_x_std"].fillna(test_data["acc_x_std"].mean(), inplace=True)


null_columns = test_data.columns[test_data.isnull().any()]
print("Columns with missing values:")
print(null_columns)


# test_data.to_parquet('test_processed.parquet')


# Step 7: Encode categorical variables
le = LabelEncoder()
train_data['gesture'] = le.fit_transform(train_data['gesture'].astype(str))
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)


## Build and submit baseline model


X = train_data.drop(['gesture', 'row_id', 'sequence_id', 'subject', 'sequence_counter'], axis=1, errors='ignore')
y = train_data['gesture']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBClassifier



model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
print(f'Validation Score: {model.score(X_val, y_val)}')


import kaggle_evaluation.cmi_inference_server

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence_pd = sequence.to_pandas()

    sequence_pd = sequence_pd.groupby('sequence_id').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)
    sequence_pd.fillna(method='bfill', inplace=True)

    numerical_cols = [col for col in sequence_pd.columns if col.startswith(('acc_', 'rot_', 'thm_', 'tof_'))]
    
    
    for col in numerical_cols:
        sequence_pd[col].fillna(sequence_pd[col].mean(), inplace=True)

    
    
    sequence_pd["acc_magnitude"] = (sequence_pd['acc_x']**2 + sequence_pd['acc_y']**2 + sequence_pd['acc_z']**2)**0.5
    sequence_pd["rot_magnitude"] = (sequence_pd['rot_x']**2 + sequence_pd['rot_y']**2 + sequence_pd['rot_z']** 2)**0.5
    sequence_pd['acc_x_diff'] = sequence_pd.groupby('sequence_id')['acc_x'].diff().fillna(0)
    sequence_pd['acc_x_mean'] = sequence_pd.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).mean().reset_index(0, drop= True)
    sequence_pd['acc_x_std'] = sequence_pd.groupby('sequence_id')['acc_x'].rolling(window = 10, min_periods = 1).std().reset_index(0, drop= True)

    
    
    for i in range(1, 6):
        tof_cols = [col for col in sequence_pd.columns if col.startswith(f'tof_{i}_')]
        sequence_pd[f'tof_{i}_mean'] = sequence_pd[tof_cols].mean(axis = 1)
        sequence_pd[f'tof_{i}_std'] = sequence_pd[tof_cols].std(axis= 1)

    sequence_pd[numerical_cols] = scaler.transform(sequence_pd[numerical_cols])
    
    # subset_test = sequence_pd[['sequence_id', 'sequence_counter', 'acc_x', 'acc_y', 'acc_z']]
    # features_test = extract_features(
    #     subset_test,
    #     column_id='sequence_id',
    #     column_sort='sequence_counter',
    #     default_fc_parameters=MinimalFCParameters()
    # )
    # features_test = features_test.reset_index().rename(columns={'index': 'sequence_id'})
    # sequence_pd = sequence_pd.merge(features_test, on='sequence_id', how='left')
    X_seq = sequence_pd.drop(['row_id', 'sequence_id', 'subject', 'sequence_counter'], axis=1, errors='ignore')
    print(X_seq.isnull().any(axis=1).sum())
    
    
    X_seq["acc_x_std"].fillna(X_seq["acc_x_std"].mean(), inplace=True)
    
    predictions = model.predict(X_seq)
    gesture = le.inverse_transform([max(set(predictions), key=predictions.tolist().count)])[0]
    return gesture

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





