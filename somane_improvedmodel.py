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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_columns = ['Compartments', 'Weight Capacity (kg)']
for col in categorical_columns + numerical_columns:
    train_data[col + '_missing'] = train_data[col].isnull().astype(int)
    test_data[col + '_missing'] = test_data[col].isnull().astype(int)
for col in categorical_columns:
    train_data[col] = train_data[col].fillna('Missing')
    test_data[col] = test_data[col].fillna('Missing')

imputer = IterativeImputer(random_state=42, max_iter=100)
train_data[numerical_columns] = imputer.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = imputer.transform(test_data[numerical_columns])





brand_price_map = train_data.groupby('Brand')['Price'].mean().to_dict()
train_data['brand_avg_price'] = train_data['Brand'].map(brand_price_map)
test_data['brand_avg_price'] = test_data['Brand'].map(brand_price_map)

global_avg = train_data['Price'].mean()
train_data['brand_avg_price'] = train_data['brand_avg_price'].fillna(global_avg)
test_data['brand_avg_price'] = test_data['brand_avg_price'].fillna(global_avg)

train_data['compartments_x_weight'] = train_data['Compartments'] * train_data['Weight Capacity (kg)']
test_data['compartments_x_weight'] = test_data['Compartments'] * test_data['Weight Capacity (kg)']


encoder = OneHotEncoder(sparse_output=False, drop='first')
X_train_cat = encoder.fit_transform(train_data[categorical_columns])
X_test_cat = encoder.transform(test_data[categorical_columns])

numerical_cols_extended = numerical_columns + ['brand_avg_price', 'compartments_x_weight']
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_data[numerical_cols_extended])
X_test_num = scaler.transform(test_data[numerical_cols_extended])

missing_cols = [col + '_missing' for col in categorical_columns + numerical_columns]
X_train_missing = train_data[missing_cols].values
X_test_missing = test_data[missing_cols].values

X_train = np.hstack((X_train_cat, X_train_num, X_train_missing))
X_test = np.hstack((X_test_cat, X_test_num, X_test_missing))
y_train = train_data['Price']

X_train_main, X_val, y_train_main, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'num_leaves': [31, 50, 70],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

lgb_model = lgb.LGBMRegressor(objective='regression', random_state=42)
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=10,  
    scoring='neg_root_mean_squared_error',  
    cv=3,  
    verbose=1,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train_main, y_train_main)
print("Best parameters found:", random_search.best_params_)
print("Best CV RMSE:", -random_search.best_score_)


best_params = random_search.best_params_
final_lgb = lgb.LGBMRegressor(objective='regression', random_state=42, **best_params)
final_lgb.fit(X_train_main, y_train_main)


xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_main, y_train_main)

lgb_val_preds = final_lgb.predict(X_val)
xgb_val_preds = xgb_model.predict(X_val)

lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_preds))
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_preds))

print(f"LightGBM RMSE on validation set: {lgb_rmse:.4f}")
print(f"XGBoost RMSE on validation set: {xgb_rmse:.4f}")

val_ensemble_preds = (lgb_val_preds + xgb_val_preds) / 2
ensemble_rmse = np.sqrt(mean_squared_error(y_val, val_ensemble_preds))
print(f"Ensemble RMSE on validation set: {ensemble_rmse:.4f}")

feature_names = (
    list(encoder.get_feature_names_out()) + 
    numerical_cols_extended + 
    missing_cols
)

lgb_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': final_lgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(lgb_importance['Feature'][:15], lgb_importance['Importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Important Features')
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.savefig('feature_importance.png')


final_lgb = lgb.LGBMRegressor(objective='regression', random_state=42, **best_params)
final_lgb.fit(X_train, y_train)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)

lgb_preds = final_lgb.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

ensemble_preds = (lgb_preds + xgb_preds) / 2

submission = pd.DataFrame({
    "id": test_data["id"],
    "Price": ensemble_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission file created!")

plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_ensemble_preds, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')







print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


print("\nMissing values in training data:")
print(train_data.isnull().sum())


print("\nSummary statistics:")
print(train_data.describe())


print("\nUnique values per column:")
print(train_data.nunique())




