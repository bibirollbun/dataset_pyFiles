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


from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 1. Load the data..
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# 2. Separate features and target
X_train = train_df.drop("rainfall", axis=1)
y_train = train_df["rainfall"]
X_test = test_df.copy()

# 3. Impute missing values
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 4. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 5. Train XGBoost Regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    n_estimators=1000,           # Adjust as needed
    learning_rate=0.05,          # Adjust as needed
    max_depth=7,                 # Adjust as needed
    subsample=0.8,               # Adjust as needed
    colsample_bytree=0.8,        # Adjust as needed
    random_state=42,
    n_jobs=-1, #use all cores available
    tree_method='hist' #performance improvement
)

model.fit(X_train_scaled, y_train,
          eval_set=[(X_train_scaled, y_train)],
          early_stopping_rounds=50,
          verbose=False)

# 6. Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# 7. Create the submission DataFrame
submission_df = pd.DataFrame({"id": test_df["id"], "rainfall": y_pred})

# 8. Save the submission DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully.")

