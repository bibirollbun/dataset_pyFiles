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

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Save test IDs
test_ids = test["id"]

# Drop 'id' column
train.drop("id", axis=1, inplace=True)
extra.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Combine all features for consistent preprocessing
combined = pd.concat([
    train.drop(columns="Price"),
    extra.drop(columns="Price"),
    test
], axis=0)

# Fill missing values
for col in combined.columns:
    if combined[col].isna().any():
        if combined[col].dtype == 'object':
            combined[col] = combined[col].fillna(combined[col].mode()[0])
        else:
            combined[col] = combined[col].fillna(combined[col].median())

# Encode categoricals
for col in combined.select_dtypes(include="object").columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
X_train_full = combined.iloc[:len(train)]
X_extra_full = combined.iloc[len(train):len(train)+len(extra)]
X_test = combined.iloc[len(train)+len(extra):]

y_train_full = train["Price"]
y_extra_full = extra["Price"]

# Split train/extra into train/val
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
X_extra, X_extra_val, y_extra, y_extra_val = train_test_split(X_extra_full, y_extra_full, test_size=0.2, random_state=42)

# Define models
model_A = HistGradientBoostingRegressor(max_iter=30, max_depth=4, random_state=1)
model_B = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=-1, random_state=1)

# Train models
model_A.fit(X_train, y_train)
model_B.fit(X_extra, y_extra)

# Evaluate RMSE
pred_val_A = model_A.predict(X_val)
pred_val_B = model_B.predict(X_extra_val)

rmse_A = mean_squared_error(y_val, pred_val_A, squared=False)
rmse_B = mean_squared_error(y_extra_val, pred_val_B, squared=False)

print(f"Model A (HistGradientBoosting) Validation RMSE: {rmse_A:.4f}")
print(f"Model B (RandomForest) Validation RMSE: {rmse_B:.4f}")

# Predict on test set
pred_A = model_A.predict(X_test)
pred_B = model_B.predict(X_test)

# Average predictions
final_preds = (pred_A + pred_B) / 2

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "Price": final_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv generated successfully.")

