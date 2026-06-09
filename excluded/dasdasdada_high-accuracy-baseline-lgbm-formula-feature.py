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
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


# --- 2. Load Data ---
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
    sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
    origin_df = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
    
    # Store test ids for submission
    test_ids = test_df['id']
    # Drop id and target
    train_df = train_df.drop('id', axis=1)
    test_df = test_df.drop('id', axis=1)
    
except FileNotFoundError:
    print("Error: Ensure data paths are correct.")
    print("Competition data should be at: /kaggle/input/playground-series-s5e10")
    print("Original dataset should be at: /kaggle/input/simulated-roads-accident-data")
    # Use placeholder data if files aren't found, to allow code to be runnable
    # In a real run, this would fail.
    train_df = pd.DataFrame() 
    test_df = pd.DataFrame()
    origin_df = pd.DataFrame()


# --- 3. Feature Engineering (The "Secret Sauce") ---

def feature_engineer(df):
    """Applies advanced feature engineering to the dataframe."""
    
    # 3.1. The "Base Risk" Formula
    # This is the key feature, derived from the original data generator
    df['base_risk'] = (
        0.3 * df["curvature"] +
        0.2 * (df["lighting"] == "night").astype(int) +
        0.1 * (df["weather"] != "clear").astype(int) +
        0.2 * (df["speed_limit"] >= 60).astype(int) +
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )
    
    # 3.2. Binary Feature Combination
    # Combine binary flags into a single categorical feature
    binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    df['binary_combo'] = 0
    for i, col in enumerate(binary_cols):
        df['binary_combo'] += df[col].astype(int) * (2**i)
        
    # 3.3. Interaction Features
    df['curvature_x_speed'] = df['curvature'] * df['speed_limit']
    df['speed_x_accidents'] = df['speed_limit'] * (df['num_reported_accidents'] + 1)
    df['weather_x_lighting'] = df['weather'].astype(str) + "_" + df['lighting'].astype(str)
    
    return df

print("Starting feature engineering...")
# Concatenate all data for consistent processing
origin_df_target = origin_df['accident_risk']
combined_df = pd.concat([
    train_df.drop('accident_risk', axis=1), 
    test_df, 
    origin_df.drop('accident_risk', axis=1)
], ignore_index=True)

# Apply feature engineering
combined_df = feature_engineer(combined_df)


# --- 4. Preprocessing (Categoricals) ---
categorical_features = [
    'road_type', 'lighting', 'weather', 
    'binary_combo', 'weather_x_lighting', 'time_of_day'
]

for col in categorical_features:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))


# --- 5. Split Data Back ---
# Separate the datasets after feature engineering
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):len(train_df) + len(test_df)]
X_origin = combined_df.iloc[len(train_df) + len(test_df):]

y = train_df['accident_risk']
y_origin = origin_df_target

# Optional: Augment training data with the original dataset
# This is a common strategy in playground competitions.
X_train_full = pd.concat([X, X_origin], ignore_index=True)
y_train_full = pd.concat([y, y_origin], ignore_index=True)


# --- 6. Model Training (LightGBM with Cross-Validation) ---

print("Starting model training...")

# LGBM parameters optimized for this task (RMSE)
# These are strong baseline parameters
lgb_params = {
    'objective': 'regression_l1', # MAE is often more robust to outliers than RMSE (L2)
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# Setup K-Fold Cross-Validation
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_predictions = np.zeros(X_train_full.shape[0])
test_predictions = np.zeros(X_test.shape[0])
models = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    model = lgb.LGBMRegressor(**lgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    
    # Predict on test set (we will average these later)
    test_preds_fold = model.predict(X_test)
    test_predictions += test_preds_fold / N_SPLITS
    models.append(model)

# Calculate OOF (Out-of-Fold) RMSE
oof_rmse = np.sqrt(mean_squared_error(y_train_full, oof_predictions))
print(f"\nOverall OOF RMSE: {oof_rmse}")


# --- 7. Create Submission File ---

# Clip predictions to be within the valid range [0, 1]
test_predictions = np.clip(test_predictions, 0, 1)

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")

