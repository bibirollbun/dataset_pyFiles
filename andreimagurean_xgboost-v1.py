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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

id_column = "id"
target = "Price"

X = train.drop([target, id_column], axis=1)
y = train[target]
X_test = test.drop(id_column, axis=1)

cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])  # assumes test set has same categories

X['Weight Capacity (kg)'] = X['Weight Capacity (kg)'].fillna(0)
X['Weight Capacity (kg)'] = X['Weight Capacity (kg)'].apply(lambda x: max(x, 0))
X_test['Weight Capacity (kg)'] = X_test['Weight Capacity (kg)'].fillna(0)
X_test['Weight Capacity (kg)'] = X_test['Weight Capacity (kg)'].apply(lambda x: max(x, 0))

#Feature engineering
X['Weight_per_Compartment'] = X['Weight Capacity (kg)'] / (X['Compartments'] + 1)
X_test['Weight_per_Compartment'] = X_test['Weight Capacity (kg)'] / (X_test['Compartments'] + 1)
rare_brands = X['Brand'].value_counts()[X['Brand'].value_counts() < 50].index
X['Brand_Grouped'] = X['Brand'].replace(rare_brands, 'Other')
X_test['Brand_Grouped'] = X_test['Brand'].replace(rare_brands, 'Other')
X['Log_Weight_Capacity'] = np.log1p(X['Weight Capacity (kg)'])
X_test['Log_Weight_Capacity'] = np.log1p(X_test['Weight Capacity (kg)'])

#Train an XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
xgb_model.fit(X, y)

y_pred = xgb_model.predict(X_test)

submission["Price"] = y_pred
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created!")

