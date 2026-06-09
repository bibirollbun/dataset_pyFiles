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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')



print(train.info())
print(train.describe())
sns.histplot(train['Price'], kde=True)
plt.title('Distribuția prețurilor')
plt.show()



train['Compartment*Weight'] = train['Compartments'] * train['Weight Capacity (kg)']
test['Compartment*Weight'] = test['Compartments'] * test['Weight Capacity (kg)']



target_col = 'Price'

X = train.drop(columns=['id', target_col])
y = train[target_col]
X_test = test.drop(columns=['id'])

# Encoding
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Imputare
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f"Baseline Random Forest RMSE: {rmse_rf:.4f}")



from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")



import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
print(f"LightGBM RMSE: {rmse_lgb:.4f}")



from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.05, 0.1]
}

xgb_model = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(f"Best Params: {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.4f}")

# Antrenăm modelul final cu cei mai buni parametri
best_xgb = XGBRegressor(**grid.best_params_, random_state=42, use_label_encoder=False, eval_metric='rmse')
best_xgb.fit(X_train, y_train)

y_pred_xgb = best_xgb.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost RMSE on validation set: {rmse_xgb:.4f}")




final_preds = lgb_model.predict(X_test)

submission = sample_submission.copy()
submission['Price'] = final_preds
submission.to_csv('submission.csv', index=False)


