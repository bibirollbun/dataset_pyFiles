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


# Step 1: Imports and Setup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")

# Config
TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
SAMPLE_SUB_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
LABEL_COL = "label"


# Step 2: Load Data
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)
sample_df = pd.read_csv(SAMPLE_SUB_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# Drop timestamp/index columns if present
if '__index_level_0__' in train_df.columns:
    train_df = train_df.drop(columns=['__index_level_0__'])

# Confirm label exists
assert LABEL_COL in train_df.columns, f"'{LABEL_COL}' column missing from train"

# Separate features and target
X = train_df.drop(columns=[LABEL_COL])
y = train_df[LABEL_COL]



# Apply PCA to reduce from ~785 columns to e.g., 100 principal components
print("Applying PCA...")
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X)
X_test_pca = pca.transform(test_df[X.columns])

print(f"PCA shape: {X_pca.shape}")


from xgboost import XGBRegressor

# Train/Validation split remains the same
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
print("Training XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'  # Faster for large datasets
)
xgb_model.fit(X_train, y_train)

# Validation Score
y_pred = xgb_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE (XGB): {mse:.5f}")



# === 9. Create Submission File ===
print("Generating submission file...")

# Load sample submission (ensure it has ID and prediction columns)
sample_df = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")

# Assign predictions
sample_df["prediction"] = test_preds

# Ensure only two required columns are present
submission_df = sample_df[["ID", "prediction"]]

# Save to Kaggle working directory
submission_df.to_csv("/kaggle/working/sample_submission.csv", index=False)

print("✅ sample_submission.csv created successfully and saved to /kaggle/working/")





