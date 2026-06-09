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
import gc

# --- 1. Load the Pre-trained Model ---
# IMPORTANT: Before running this notebook, you must add the output of your training notebook as a data source.
# 1. Go to your notebook viewer for THIS submission notebook.
# 2. Click "+ Add data" in the top-right.
# 3. Select the "Notebook Output Files" tab.
# 4. Find your training notebook and click "Add".
# The model will then be available at a path like the one below.
# You MUST change 'your-training-notebook-name' to the actual URL name of your training notebook.
model_path = '/kaggle/input/notebook5f77c8150b/lgbm_final_model.txt'  # <-- CHANGE THIS PATH
model = lgb.Booster(model_file=model_path)

# --- 2. Re-create the EXACT SAME Feature Engineering Function ---
# This must be identical to the one used for training.
def enhanced_feature_engineering_mem_safe(df):
    """V2.1 of feature engineering, slightly trimmed for memory."""
    df_out = df.copy()
    # Log transform
    skewed_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for col in skewed_features:
        df_out[f'{col}_log'] = np.log1p(df_out[col])
    # Ratios
    df_out['order_book_imbalance'] = (df_out['bid_qty'] - df_out['ask_qty']) / (df_out['bid_qty'] + df_out['ask_qty'])
    df_out['trade_imbalance'] = (df_out['buy_qty'] - df_out['sell_qty']) / (df_out['buy_qty'] + df_out['sell_qty'])
    # Expanded Rolling Windows
    windows = [5, 10, 30, 60]
    features_to_roll = ['label', 'X719']
    for window in windows:
        for feat in features_to_roll:
            shifted_feat = df_out[feat].shift(1)
            df_out[f'{feat}_roll_std_{window}'] = shifted_feat.rolling(window=window).std()
            df_out[f'{feat}_roll_mean_{window}'] = shifted_feat.rolling(window=window).mean()
    # Lag the most important raw X features
    important_x_features = ['X719', 'X235', 'X718']
    for lag in [1, 2, 5]:
        for feat in important_x_features:
            df_out[f'{feat}_lag_{lag}'] = df_out[feat].shift(lag)
    # Momentum Features
    momentum_features = ['label', 'X719', 'trade_imbalance']
    for lag in [1, 5]:
        for feat in momentum_features:
            df_out[f'{feat}_mom_{lag}'] = df_out[feat] - df_out[feat].shift(lag)
    # The label in the test set is always 0, so rolling features on it will be 0, which is fine.
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    return df_out

# --- 3. Load and Process Test Data ---
# Load the test data from the Parquet file
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet', engine='pyarrow')

# Apply feature engineering
df_proc = enhanced_feature_engineering_mem_safe(test_df)

# Get the feature names from the trained model
features = model.feature_name()

# Select only the features the model was trained on
X_test = df_proc[features]

# Fill any NaNs that might still exist (e.g., from lags or rolling features)
X_test = X_test.fillna(0)

# --- 4. Make Predictions ---
predictions = model.predict(X_test)

# --- 5. Create Submission File ---
# Load the sample submission file
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Assign predictions to the submission file
# Assuming the sample submission expects a 'prediction' column (adjust if the column name is 'label')
sample_submission['prediction'] = predictions

# Save the submission file
sample_submission.to_csv('submission.csv', index=False)

# --- 6. Clean Up Memory ---
del df_proc, X_test, predictions
gc.collect()

print("Submission complete.")

