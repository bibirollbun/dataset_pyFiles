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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


X = train.drop(['Price', 'id'], axis=1)
y = train['Price']
X_test = test.drop('id', axis=1)

cat_cols = X.select_dtypes(include="object").columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le  # Store for later use if needed

X['Weight_per_Compartment'] = X['Weight Capacity (kg)'] / (X['Compartments'] + 1)
X_test['Weight_per_Compartment'] = X_test['Weight Capacity (kg)'] / (X_test['Compartments'] + 1)

rare_brands = X['Brand'].value_counts()[X['Brand'].value_counts() < 50].index
X['Brand_Grouped'] = X['Brand'].replace(rare_brands, 'Other')
X_test['Brand_Grouped'] = X_test['Brand'].replace(rare_brands, 'Other')

brand_le = LabelEncoder()
X['Brand_Grouped'] = brand_le.fit_transform(X['Brand_Grouped'])
X_test['Brand_Grouped'] = brand_le.transform(X_test['Brand_Grouped'])

X['Log_Weight_Capacity'] = np.log1p(X['Weight Capacity (kg)'])
X_test['Log_Weight_Capacity'] = np.log1p(X_test['Weight Capacity (kg)'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = xgb.XGBRegressor(random_state=42)
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search_xgb.fit(X_train, y_train)

print("Best XGBoost Parameters:", grid_search_xgb.best_params_)

best_xgb_model = grid_search_xgb.best_estimator_
y_pred_val = best_xgb_model.predict(X_val)
rmse_best = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"ğŸ“� Best XGBoost RMSE after tuning: {rmse_best:.4f}")

final_model = xgb.XGBRegressor(**grid_search_xgb.best_params_, random_state=42)
final_model.fit(X, y)

joblib.dump(final_model, "final_xgboost_model.pkl")

test_preds = final_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "Price": test_preds
})
submission.to_csv("submission2.csv", index=False)

print("âœ… Model retrained, saved, and submission2.csv created.")


