# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train["public_road"].unique()


train.head()


test.head()


print("Train :",train.shape)
print("Test :", test.shape)


train.columns


train.isnull().sum()


test.isnull().sum()


X = train.drop(['id','accident_risk'], axis =1)
y = train["accident_risk"]

X_test = test.drop(['id'], axis =1)


cat_col = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_col = X.select_dtypes(exclude=['object', 'bool']).columns.tolist()


print("Categorical columns:", cat_col)
print("Numeric columns:", num_col)


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col),
    ('num', StandardScaler(), num_col)
])



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



rf_model = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=800,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])



rf_model.fit(X_train, y_train)



y_pred = rf_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)



lgb_model = Pipeline([
    ('preprocess', preprocessor),
    ('model', LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])



from sklearn.model_selection import cross_val_score, KFold
import numpy as np

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

print("Cross-Validation RMSE:", np.mean(-scores))



lgb_model.fit(X_train, y_train)



y_pred = lgb_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE:", rmse)



xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist'  # faster on CPUs
    ))
])




xgb_model.fit(X_train, y_train)


val_preds = xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"✅ Validation RMSE: {rmse:.6f}")


from sklearn.model_selection import cross_val_score, KFold
import numpy as np

cv = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

print("Cross-Validation for XGB Model RMSE:", np.mean(-xgb_scores))



from sklearn.model_selection import cross_val_score, KFold
import numpy as np

cv = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_scores = cross_val_score(lgb_model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

print("Cross-Validation for LGB Model RMSE:", np.mean(-lgb_scores))



test_preds = xgb_model.predict(test.drop(['id'], axis=1))

# Clip predictions to [0, 1] since accident risk is probability-like
test_preds = np.clip(test_preds, 0, 1)


submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_preds
})


submission.to_csv('submission_xgboost.csv', index=False)
print("✅ submission_xgboost.csv file created successfully!")
submission.head() 


import joblib

joblib.dump(lgb_model, "model_lgbm.pkl")
joblib.dump(xgb_model, "model_xgb.pkl")
joblib.dump(rf_model, "model_rf.pkl")
import joblib

joblib.dump(model, "xgb_model_repacked.pkl")




train.columns





