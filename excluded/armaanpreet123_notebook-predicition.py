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


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# **1. Load Data**
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# **2. Define Important Columns**
target_col = "rainfall"  # Target variable
id_col = "id"  # ID column

# **3. Drop Target & ID from Features**
X = train.drop(columns=[id_col, target_col], errors="ignore")
y = train[target_col]

X_test = test.drop(columns=[id_col], errors="ignore")

# **4. Ensure Test Features Match Train Features**
X_test = X_test[X.columns]  # Ensure test columns match training features

# **5. Train-Test Split**
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# **6. Initialize CatBoost Model**
model = CatBoostRegressor(
    iterations=1000, 
    learning_rate=0.05, 
    depth=6, 
    l2_leaf_reg=3, 
    loss_function="RMSE",
    early_stopping_rounds=50,
    silent=True  # Suppresses logs for clarity
)

# **7. Train Model**
model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    verbose=100
)

# **8. Evaluate Model**
y_valid_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_valid_pred)
print(f"Validation MSE: {mse:.4f}")

# **9. Debugging Feature Names**
print("Features in Training:", X.columns.tolist())
print("Features in X_test:", X_test.columns.tolist())

# **10. Make Predictions**
y_test_pred = model.predict(X_test)

# **11. Save Submission**
submission = pd.DataFrame({id_col: test["id"], target_col: y_test_pred})
submission.to_csv("submission.csv", index=False)
print("✅ CatBoost model submission saved successfully!")





