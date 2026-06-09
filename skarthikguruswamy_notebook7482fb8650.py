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
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("Current working dir:", os.getcwd())

# Kaggle paths (as provided)
train_path = '/kaggle/input/playground-series-s5e9/train.csv'
test_path  = '/kaggle/input/playground-series-s5e9/test.csv'

# If the Kaggle paths don't exist in this runtime, fallback to local /mnt/data if user placed files there
if not os.path.exists(train_path):
    alt_train = '/mnt/data/train.csv'
    if os.path.exists(alt_train):
        train_path = alt_train

if not os.path.exists(test_path):
    alt_test = '/mnt/data/test.csv'
    if os.path.exists(alt_test):
        test_path = alt_test

print("Train path:", train_path)
print("Test path:", test_path)



# Try to load train; if not present create a small synthetic dataset (so notebook remains runnable)
if os.path.exists(train_path):
    df = pd.read_csv(train_path)
    print("Loaded train.csv with shape:", df.shape)
else:
    print("train.csv not found at expected path. Creating synthetic fallback dataset.")
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        'RhythmScore': rng.normal(0.5, 0.15, n).clip(0,1),
        'AudioLoudness': rng.normal(65, 5, n).clip(0,200),
        'VocalContent': np.abs(rng.normal(0.3, 0.25, n)),
        'AcousticQuality': np.abs(rng.normal(0.6, 0.2, n)),
        'InstrumentalScore': np.abs(rng.exponential(0.8, n)).clip(0,1),
        'LivePerformanceLikelihood': rng.uniform(0,1,n),
        'MoodScore': rng.normal(0.5,0.2,n).clip(0,1),
        'TrackDurationMs': rng.integers(150000, 300000, n),
        'Energy': rng.normal(0.6,0.15,n).clip(0,1),
        'BeatsPerMinute': rng.integers(60,180,n)
    })
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    print("Created synthetic df with shape:", df.shape)

# Show columns and a few rows
display(df.head())

# Check for required columns
required = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore',             'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in train dataset: {missing}")


# Separate predictors and target
target_col = 'BeatsPerMinute'
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Features to transform due to skew (based on prior analysis)
skewed = ['VocalContent', 'AcousticQuality', 'InstrumentalScore']

# Check positivity for log1p; if any non-positive values, use PowerTransformer (Yeo-Johnson) as fallback
from sklearn.preprocessing import PowerTransformer
for col in skewed:
    if (X[col] <= 0).any():
        print(f"Column {col} has non-positive values; will use Yeo-Johnson PowerTransformer instead of log1p.")
        use_power = True
        break
else:
    use_power = False

if not use_power:
    for col in skewed:
        X[col] = np.log1p(X[col])
    print('Applied log1p to:', skewed)
else:
    pt = PowerTransformer(method='yeo-johnson')
    X[skewed] = pt.fit_transform(X[skewed])
    print('Applied Yeo-Johnson to:', skewed)

# Optional: print skewness before/after for quick check
orig_skew = df[skewed].skew()
new_skew = X[skewed].skew()
print("Original skewness for skewed cols:\n", orig_skew)
print("New skewness after transform:\n", new_skew)



# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# We won't scale for XGBoost, but show how to if you want consistent scaling across pipelines
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)



# Train a basic XGBoost regressor with early stopping
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',  # fast and deterministic in many environments
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)

# Predict and compute RMSE on validation
val_preds = model.predict(X_val)
val_rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

base_estimators = [
    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbosity=0)),
    ('lgb', LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42))
]

stack = StackingRegressor(estimators=base_estimators, final_estimator=Ridge(alpha=1.0), n_jobs=-1, passthrough=False)
stack.fit(X_train, y_train)

preds_val = stack.predict(X_val)
rmse_stack = mean_squared_error(y_val, preds_val, squared=False)
print("Stacking val RMSE:", rmse_stack)




import joblib

# --- Save models ---
# Save the XGBoost regressor (native XGB format, portable across languages)
model.save_model("xgb_model.json")

# Save the stacking regressor (use joblib, since it's sklearn-based)
joblib.dump(stack, "stacking_model.pkl")

print("Models saved successfully.")


# If a test file exists, try to load and evaluate if target present, otherwise produce predictions
output_path = '/kaggle/working/submission.csv'

if os.path.exists(test_path):
    df_test = pd.read_csv(test_path)
    print("Loaded test.csv with shape:", df_test.shape)
    # Ensure required predictor columns exist
    missing_test = [c for c in X.columns if c not in df_test.columns]
    if missing_test:
        raise ValueError(f"Test file is missing predictor columns: {missing_test}")
    X_test = df_test[X.columns].copy()
    # Apply same transformations to skewed cols
    # if not use_power:
    #     X_test[skewed] = np.log1p(X_test[skewed])
    # else:
    #     X_test[skewed] = pt.transform(X_test[skewed])
    # If test has target, evaluate RMSE
    if 'BeatsPerMinute' in df_test.columns:
        y_test = df_test['BeatsPerMinute']
        test_preds = stack.predict(X_test)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
        print(f"Test RMSE (using test.csv target): {test_rmse:.4f}")
    else:
        # Save predictions for submission / inspection
        preds = stack.predict(X_test)
        df_sub = pd.DataFrame({'id': df_test['id'],
                               'BeatsPerMinute': preds})
        df_sub.to_csv(output_path, index=False)
        print(f"Test file had no target: wrote predictions to {output_path}")
else:
    print('No test.csv found at provided path; skipping test evaluation.')





