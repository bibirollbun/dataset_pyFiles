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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import lightgbm as lgbm


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

print("Train shape:", train.shape)
print("Missing values in train:", train.isnull().sum())


train = train.copy()
test = test.copy()

#fill missing numerical values:
weight_median = train["Weight Capacity (kg)"].median()
train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(weight_median)
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(weight_median)

#extract numeric part of size:
train['Size_Numeric'] = pd.to_numeric(train['Size'].str.extract('(\d+)')[0], errors='coerce')
test['Size_Numeric'] = pd.to_numeric(test['Size'].str.extract('(\d+)')[0], errors='coerce')
size_median = train['Size_Numeric'].median()
train['Size_Numeric'] = train['Size_Numeric'].fillna(size_median)
test['Size_Numeric'] = test['Size_Numeric'].fillna(size_median)

#create brand-based features:
brand_avg_price = train.groupby('Brand')['Price'].mean()
brand_count = train.groupby('Brand').size()
train['Brand_Avg_Price'] = train['Brand'].map(brand_avg_price)
train['Brand_Popularity'] = train['Brand'].map(brand_count)
test['Brand_Avg_Price'] = test['Brand'].map(brand_avg_price).fillna(train['Price'].mean())
test['Brand_Popularity'] = test['Brand'].map(brand_count).fillna(1)

#log-transform weight capacity:
train['Weight_Capacity_Log'] = np.log1p(train['Weight Capacity (kg)'].clip(lower=0))
test['Weight_Capacity_Log'] = np.log1p(test['Weight Capacity (kg)'].clip(lower=0))

#convert boolean to numeric:
for column in ['Laptop Compartment', 'Waterproof']:
    train[column] = train[column].map({'Yes': 1, 'No': 0, None: 0.5})
    test[column] = test[column].map({'Yes': 1, 'No': 0, None: 0.5})
    
#fill categorical NAs:
categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']
for column in categorical_cols:
    mode_val = train[column].mode()[0]
    train[column] = train[column].fillna(mode_val)
    test[column] = test[column].fillna(mode_val)


A = train.drop(["id", "Price"], axis=1)
B = train["Price"]
A_test = test.drop(["id"], axis=1)
A_train, A_val, B_train, B_val = train_test_split(A, B, test_size=0.2, random_state=42)


numerical_cols = ['Weight Capacity (kg)', 'Weight_Capacity_Log', 'Size_Numeric', 
                  'Brand_Avg_Price', 'Brand_Popularity',
                  'Laptop Compartment', 'Waterproof', 'Compartments']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill NaNs first
    ('scaler', RobustScaler())
])

numerical_cols = [col for col in numerical_cols if not A_train[col].isna().all()]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


A = train.drop(['id', 'Price'], axis=1)
B = np.log1p(train['Price'])  # log(Price + 1)
A_test = test.drop(['id'], axis=1)

#split data
A_train, A_val, B_train, B_val = train_test_split(A, B, test_size=0.2, random_state=42)

lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', lgbm.LGBMRegressor(
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42))
])

print("Training LightGBM model...")
lgbm_pipeline.fit(A_train, B_train)

#evaluate LightGBM
lgbm_val_preds = np.expm1(lgbm_pipeline.predict(A_val))
lgbm_rmse = mean_squared_error(np.expm1(B_val), lgbm_val_preds, squared=False)
print(f"LightGBM Validation Root Mean Squared Error: {lgbm_rmse}")


A_train_num = A_train[numerical_cols].copy()
A_val_num = A_val[numerical_cols].copy()
A_test_num = A_test[numerical_cols].copy()

#handle missing values
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')

A_train_num = pd.DataFrame(num_imputer.fit_transform(A_train_num), columns=numerical_cols)
A_val_num = pd.DataFrame(num_imputer.transform(A_val_num), columns=numerical_cols)
A_test_num = pd.DataFrame(num_imputer.transform(A_test_num), columns=numerical_cols)

#train and evaluate
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(A_train_num, B_train)

rf_val_preds = np.expm1(rf_model.predict(A_val_num))
rf_rmse = mean_squared_error(np.expm1(B_val), rf_val_preds, squared=False)
print(f"Random Forest Validation RMSE: {rf_rmse}")


print("Generating predictions...")
lgbm_preds = np.expm1(lgbm_pipeline.predict(A_test))
rf_preds = np.expm1(rf_model.predict(A_test_num))

# Weighted average ensemble
if lgbm_rmse < rf_rmse:
    final_preds = 0.8 * lgbm_preds + 0.2 * rf_preds
    print("Using weighted ensemble (LightGBM stronger)")
else:
    final_preds = 0.5 * lgbm_preds + 0.5 * rf_preds
    print("Using equally weighted ensemble")


submission["Price"] = final_preds
submission.to_csv("improved_submission.csv", index=False)
print("Submission file created: improved_submission.csv")

