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


train = pd.read_csv("/kaggle/input/terrain-prices-reggression/train.csv")
test = pd.read_csv("/kaggle/input/terrain-prices-reggression/test.csv")


train.head()


test.head()


train = pd.get_dummies(train, columns=["land_use", "zoning_code"], drop_first= True)
test = pd.get_dummies(test, columns=["land_use", "zoning_code"], drop_first= True)


freq_map = train["location_type"].value_counts(normalize=True)
train["location_type_fe"] = train["location_type"].map(freq_map)
test["location_type_fe"] = test["location_type"].map(freq_map).fillna(0)

# Drop original column
train.drop(columns=["location_type"], inplace=True)
test.drop(columns=["location_type"], inplace=True)


train.shape


test.shape


train = train.drop(columns=["id"])
test = test.drop(columns=["id"])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


X = train.drop(columns=["target"])
y = train["target"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


y_pred = rf.predict(X_val)

r2 = r2_score(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)

print(f"Validation R²:   {r2:.4f}")
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation MAE:  {mae:.4f}")


test_preds = rf.predict(test)

ids = [f"id_{i}" for i in range(len(test_preds))]

submission = pd.DataFrame({
    "id": ids,
    "target": test_preds
})

submission.to_csv("submission_file.csv", index=False)

print("submission.csv saved!")





