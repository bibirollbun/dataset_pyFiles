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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train


train.dtypes


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test


test.columns


scaler = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
ohe = ['road_type', 'lighting', 'weather', 'time_of_day']


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


X = train.drop(['accident_risk', 'id'], axis=1)
y = train['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), scaler),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ohe)
    ],
    remainder='passthrough' 
)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_transformed, y_train)


y_pred = model.predict(X_test_transformed)


r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")


submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
submission 


from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=100
)
cat_model.fit(X_train_transformed, y_train)



y_pred1 = cat_model.predict(X_test_transformed)


r2 = r2_score(y_test, y_pred1)
mae = mean_absolute_error(y_test, y_pred1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred1))

print(f"R² Score: {r2:.5f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")


from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_transformed, y_train)


y_pred2 = xgb_model.predict(X_test_transformed)


r2 = r2_score(y_test, y_pred2)


print(f"R² Score: {r2:.5f}")


from xgboost import XGBRegressor

xgb_model1 = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model1.fit(X_train_transformed, y_train)


y_pred2 = xgb_model1.predict(X_test_transformed)

r2 = r2_score(y_test, y_pred2)

print(f"R² Score: {r2:.5f}")


from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01, random_state=42, max_iter=10000)
lasso_model.fit(X_train_transformed, y_train)


y_pred2 = lasso_model.predict(X_test_transformed)

r2 = r2_score(y_test, y_pred2)

print(f"R² Score: {r2:.5f}")


from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train_transformed, y_train)


y_pred2 = lgb_model.predict(X_test_transformed)

r2 = r2_score(y_test, y_pred2)

print(f"R² Score: {r2:.5f}")


X_test = test.drop(columns=['id'])

X_test_prepared = preprocessor.transform(X_test)

test_predictions = xgb_model1.predict(X_test_prepared)


submission = submission.copy()

submission['accident_risk'] = test_predictions

submission.to_csv('submission.csv', index=False)

submission.head()

