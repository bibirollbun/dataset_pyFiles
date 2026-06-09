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
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

median_weight = train["Weight Capacity (kg)"].median()
train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(median_weight)
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(median_weight)

cat_columns = ["Brand", "Material", "Size", "Laptop Compartment",
               "Waterproof", "Style", "Color"]

for col in cat_columns:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")
    le = LabelEncoder()
    le.fit(list(train[col].astype(str)) + list(test[col].astype(str)))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

train["Brand_Material"] = train["Brand"] * train["Material"]
test["Brand_Material"] = test["Brand"] * test["Material"]

train["Compartments_per_Capacity"] = train["Compartments"] / (train["Weight Capacity (kg)"] + 1e-5)
test["Compartments_per_Capacity"] = test["Compartments"] / (test["Weight Capacity (kg)"] + 1e-5)

X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])

# Împărțire în train și validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.08,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=15, importance_type='gain', height=0.6)
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.2f}")

submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
submission["Price"] = model.predict(X_test)
submission.to_csv("submission.csv", index=False)


