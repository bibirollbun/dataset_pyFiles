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



import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

try:
    df_train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
    df_val = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/val.csv')
    df_test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("ERROR: Could not find data files. Make sure the Kaggle input path is correct.")
    exit()

df_train.columns = df_train.columns.str.lower()
df_val.columns = df_val.columns.str.lower()
df_test.columns = df_test.columns.str.lower()
print("\nAll column names converted to lowercase for consistency.")
print(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}")

def feature_engineer(df):
    df_out = df.copy()
    df_out['corners_per_lap'].fillna(0, inplace=True)
    df_out['circuit_length_km'].fillna(1, inplace=True)
    df_out['corners_per_km'] = df_out['corners_per_lap'] / df_out['circuit_length_km']
    df_out['temp_difference'] = df_out['track_temperature_celsius'] - df_out['ambient_temperature_celsius']
    epsilon = 1e-6
    df_out['win_rate'] = df_out['wins'] / (df_out['starts'] + epsilon)
    df_out['podium_rate'] = df_out['podiums'] / (df_out['starts'] + epsilon)
    df_out['finish_rate'] = df_out['finishes'] / (df_out['starts'] + epsilon)
    df_out.replace([np.inf, -np.inf], 0, inplace=True)
    df_out.fillna(0, inplace=True)
    return df_out

print("\nFeature engineering function defined.")

print("\nStarting data preprocessing...")

TARGET = 'lap_time_seconds'

combined_df = pd.concat([df_train.drop(TARGET, axis=1), df_val.drop(TARGET, axis=1), df_test], ignore_index=True)

cols_to_drop = [
    'rider_name', 'team_name', 'bike_name', 'circuit_name',
    'track', 'air', 'ground', 'position', 'points', 'shortname', 'year_x', 'sequence'
]
combined_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} columns.")

combined_df = feature_engineer(combined_df)
print("Feature engineering applied.")

numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()
categorical_features = combined_df.select_dtypes(exclude=np.number).columns.tolist()

numerical_features.remove('unique id')

imputer_num = SimpleImputer(strategy='median')
combined_df[numerical_features] = imputer_num.fit_transform(combined_df[numerical_features])

imputer_cat = SimpleImputer(strategy='most_frequent')
combined_df[categorical_features] = imputer_cat.fit_transform(combined_df[categorical_features])
print("Missing values imputed.")

for col in categorical_features:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
print("Categorical features label-encoded.")

scaler = StandardScaler()
combined_df[numerical_features] = scaler.fit_transform(combined_df[numerical_features])
print("Numerical features scaled.")

train_processed = combined_df.iloc[:len(df_train)]
val_processed = combined_df.iloc[len(df_train):len(df_train) + len(df_val)]
test_processed = combined_df.iloc[len(df_train) + len(df_val):]

test_ids = test_processed['unique id']
train_processed.drop(columns=['unique id'], inplace=True)
val_processed.drop(columns=['unique id'], inplace=True)
test_processed.drop(columns=['unique id'], inplace=True)

y_train = df_train[TARGET]
y_val = df_val[TARGET]

print("\nPreprocessing complete. Final shapes:")
print(f"X_train: {train_processed.shape}, y_train: {y_train.shape}")
print(f"X_val: {val_processed.shape}, y_val: {y_val.shape}")
print(f"X_test: {test_processed.shape}")
print("\n--- Training LightGBM Model ---")

print("\n--- Training XGBoost Model ---")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=15,
    colsample_bytree=0.75,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model.fit(
    train_processed, y_train,
    eval_set=[(val_processed, y_val)],
    verbose=False
)

y_pred_xgb_val = xgb_model.predict(val_processed)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb_val))
print(f"ðŸš€ Single XGBoost Model - Validation RMSE: {rmse_xgb:.5f}")

print("\n--- Generating Final Predictions for Submission ---")
preds_test_xgb = xgb_model.predict(test_processed)
final_predictions = preds_test_xgb
submission_df = pd.DataFrame({
    'Unique ID': test_ids.astype(int), 
    'Lap_Time_Seconds': final_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("\nâœ… Submission file 'submission.csv' created successfully!")
print("Top 5 rows of submission file:")
print(submission_df.head())


