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

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load data
extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Save test IDs
test_ids = test["id"]

# Drop ID columns
extra.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

# Combine for preprocessing
combined = pd.concat([extra.drop(columns="Price"), test], axis=0)

# Fill missing values
for col in combined.columns:
    if combined[col].isna().any():
        if combined[col].dtype == 'object':
            combined[col] = combined[col].fillna(combined[col].mode()[0])
        else:
            combined[col] = combined[col].fillna(combined[col].median())

# Encode categorical columns
for col in combined.select_dtypes(include="object").columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Split back
X_extra_full = combined.iloc[:len(extra)]
X_test = combined.iloc[len(extra):]
y_extra_full = extra["Price"]

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X_extra_full, y_extra_full, test_size=0.2, random_state=42)

# Define models
model_ridge = Ridge(alpha=1.0)
model_mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, early_stopping=True, random_state=1)

# Train models
model_ridge.fit(X_train, y_train)
model_mlp.fit(X_train, y_train)

# Validation RMSE
val_pred_ridge = model_ridge.predict(X_val)
val_pred_mlp = model_mlp.predict(X_val)

rmse_ridge = mean_squared_error(y_val, val_pred_ridge, squared=False)
rmse_mlp = mean_squared_error(y_val, val_pred_mlp, squared=False)

print(f"Ridge Validation RMSE: {rmse_ridge:.4f}")
print(f"MLP Validation RMSE:   {rmse_mlp:.4f}")

# Predict on test
pred_ridge = model_ridge.predict(X_test)
pred_mlp = model_mlp.predict(X_test)
final_preds = (pred_ridge + pred_mlp) / 2

# Submission
submission = pd.DataFrame({
    "id": test_ids,
    "Price": final_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv generated.")

