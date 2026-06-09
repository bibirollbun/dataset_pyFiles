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
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


competition_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.head()


df.drop('id', axis=1).info()
df.drop('id', axis=1).describe()


categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"Unique values in {col}: {df[col].unique()}")


df.isnull().sum()


X = df.drop(['Price', 'id'], axis=1)

y = df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train.astype({col: 'category' for col in X_train.select_dtypes(include=['object']).columns})
X_val = X_val.astype({col: 'category' for col in X_val.select_dtypes(include=['object']).columns})
X_test = X_test.astype({col: 'category' for col in X_test.select_dtypes(include=['object']).columns})


# XGBoost Model
xgb_model = XGBRegressor(enable_categorical=True)
xgb_model.fit(X_train, y_train)

# Predict on the validation set
xgb_val_preds = xgb_model.predict(X_val)
xgb_val_rmse = mean_squared_error(y_val, xgb_val_preds, squared=False)

# Predict on the test set
xgb_test_preds = xgb_model.predict(X_test)
xgb_test_rmse = mean_squared_error(y_test, xgb_test_preds, squared=False)


print("XGBoost Validation RMSE:", xgb_val_rmse)
print("XGBoost Test RMSE:", xgb_test_rmse)


# LightGBM Model
lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Predict on the validation set
lgb_val_preds = lgb_model.predict(X_val)
lgb_val_rmse = mean_squared_error(y_val, lgb_val_preds, squared=False)

# Predict on the test set
lgb_test_preds = lgb_model.predict(X_test)
lgb_test_rmse = mean_squared_error(y_test, lgb_test_preds, squared=False)

print("LightGBM Validation RMSE:", lgb_val_rmse)
print("LightGBM Test RMSE:", lgb_test_rmse)


# CatBoost Model
cat_features = list(set(X_train.select_dtypes(include=['category']).columns) & 
                    set(X_val.select_dtypes(include=['category']).columns) & 
                    set(X_test.select_dtypes(include=['category']).columns))

X_train = X_train.astype(str)
X_val = X_val.astype(str)
X_test = X_test.astype(str)

cat_model = CatBoostRegressor(verbose=0)
cat_model.fit(X_train, y_train, cat_features=cat_features)

# Predict on Validation data
cat_preds = cat_model.predict(X_val)
print("CatBoost RMSE (val):", mean_squared_error(y_val, cat_preds, squared=False))

# Predict on test data
cat_preds = cat_model.predict(X_test)
print("CatBoost RMSE (test):", mean_squared_error(y_test, cat_preds, squared=False))



competition_df.isnull().sum()


df.isnull().any(axis=1).sum()


df.duplicated().sum()


df_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print('Extra train dataset no. of All the Records:', df_ex.shape[0])
print('Extra train dataset no. of Null Records only:', df_ex.isnull().any(axis=1).sum())
df_ex.isnull().sum()


df_ex.duplicated().sum()


df[df['Compartments'] % 1 != 0]


pip install ydata-profiling


profile = ProfileReport(df.drop('id', axis=1), title='Pandas Profile', minimal=True, progress_bar=False)
profile.to_notebook_iframe()


df_d = df.dropna()


df_d.drop('id', axis=1).hist(figsize=(12, 10))
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_d.drop('id', axis=1))
plt.ylim(bottom=0, top=180)  
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_d, x='Brand', y='Price')
plt.xticks(rotation=90)
plt.title('Impact of Brand on Price')
plt.show()

categorical_columns = df_d.drop('id', axis=1).select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_d, x=col)
    plt.title(f'Count Plot of {col}')
    plt.show()


numerical_columns = df_d.select_dtypes(include=['float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


df_d_e = pd.get_dummies(df_d, columns=['Brand', 'Material','Laptop Compartment', 
                                                   'Waterproof', 'Style', 'Color'], drop_first=True)


df_d_e['Size'] = df_d_e['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})


X = df_d_e.drop(['Price', 'id'], axis=1)

y = df_d_e['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_lr_val = lr_model.predict(X_val)
# Predict on the test set
y_pred_lr_test = lr_model.predict(X_test)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_dt_val = dt_model.predict(X_val)
# Predict on the test set
y_pred_dt_test = dt_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the validation set
y_pred_rf_val = rf_model.predict(X_val)
# Predict on the test set
y_pred_rf_test = rf_model.predict(X_test)

# Evaluate the models on the validation set
print("Validation Set Evaluation:")

# Linear Regression Evaluation
lr_rmse_val = mean_squared_error(y_val, y_pred_lr_val, squared=False)

# Decision Tree Evaluation
dt_rmse_val = mean_squared_error(y_val, y_pred_dt_val, squared=False)

# Random Forest Evaluation
rf_rmse_val = mean_squared_error(y_val, y_pred_rf_val, squared=False)

# Print out the validation set evaluation results
print(f"Linear Regression Validation RMSE: {lr_rmse_val}")

print(f"Decision Tree Validation MSE: {dt_rmse_val}")

print(f"Random Forest Validation MSE: {rf_rmse_val}")

# Evaluate the models on the test set
print("Test Set Evaluation:")

# Linear Regression Evaluation
lr_rmse_test = mean_squared_error(y_test, y_pred_lr_test, squared=False)

# Decision Tree Evaluation
dt_rmse_test = mean_squared_error(y_test, y_pred_dt_test, squared=False)

# Random Forest Evaluation
rf_rmse_test = mean_squared_error(y_test, y_pred_rf_test, squared=False)

# Print out the test set evaluation results
print(f"Linear Regression Test RMSE: {lr_rmse_test}")

print(f"Decision Tree Test RMSE: {dt_rmse_test}")

print(f"Random Forest Test RMSE: {rf_rmse_test}")


print('Target mean:', df['Price'].mean())
print('Target median:', df['Price'].median())


combined_df = pd.concat([df, df_ex], axis=0)


X = combined_df.drop(['Price', 'id'], axis=1)

y = combined_df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train.astype({col: 'category' for col in X_train.select_dtypes(include=['object']).columns})
X_val = X_val.astype({col: 'category' for col in X_val.select_dtypes(include=['object']).columns})
X_test = X_test.astype({col: 'category' for col in X_test.select_dtypes(include=['object']).columns})


# XGBoost Model
xgb_model = XGBRegressor(enable_categorical=True)
xgb_model.fit(X_train, y_train)

# Predict on the validation set
xgb_val_preds = xgb_model.predict(X_val)
xgb_val_rmse = mean_squared_error(y_val, xgb_val_preds, squared=False)

# Predict on the test set
xgb_test_preds = xgb_model.predict(X_test)
xgb_test_rmse = mean_squared_error(y_test, xgb_test_preds, squared=False)


print("XGBoost Validation RMSE:", xgb_val_rmse)
print("XGBoost Test RMSE:", xgb_test_rmse)


# LightGBM Model
lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Predict on the validation set
lgb_val_preds = lgb_model.predict(X_val)
lgb_val_rmse = mean_squared_error(y_val, lgb_val_preds, squared=False)

# Predict on the test set
lgb_test_preds = lgb_model.predict(X_test)
lgb_test_rmse = mean_squared_error(y_test, lgb_test_preds, squared=False)

print("LightGBM Validation RMSE:", lgb_val_rmse)
print("LightGBM Test RMSE:", lgb_test_rmse)


combined_df_d = combined_df.dropna()


# # Encoding
# combined_df_d_e = pd.get_dummies(combined_df_d, columns=['Brand', 'Material','Laptop Compartment', 
#                                                    'Waterproof', 'Style', 'Color'], drop_first=True)
# combined_df_d_e['Size'] = combined_df_d_e['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3})

# X = combined_df_d_e.drop(['Price', 'id'], axis=1)

# y = combined_df_d_e['Price']

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# # Linear Regression
# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)

# # Predict on the validation set
# y_pred_lr_val = lr_model.predict(X_val)
# # Predict on the test set
# y_pred_lr_test = lr_model.predict(X_test)

# # Decision Tree Regressor
# dt_model = DecisionTreeRegressor(random_state=42)
# dt_model.fit(X_train, y_train)

# # Predict on the validation set
# y_pred_dt_val = dt_model.predict(X_val)
# # Predict on the test set
# y_pred_dt_test = dt_model.predict(X_test)

# # Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Predict on the validation set
# y_pred_rf_val = rf_model.predict(X_val)
# # Predict on the test set
# y_pred_rf_test = rf_model.predict(X_test)

# # Evaluate the models on the validation set
# print("Validation Set Evaluation:")

# # Linear Regression Evaluation
# lr_rmse_val = mean_squared_error(y_val, y_pred_lr_val, squared=False)

# # Decision Tree Evaluation
# dt_rmse_val = mean_squared_error(y_val, y_pred_dt_val, squared=False)

# # Random Forest Evaluation
# rf_rmse_val = mean_squared_error(y_val, y_pred_rf_val, squared=False)

# # Print out the validation set evaluation results
# print(f"Linear Regression Validation RMSE: {lr_rmse_val}")

# print(f"Decision Tree Validation MSE: {dt_rmse_val}")

# print(f"Random Forest Validation MSE: {rf_rmse_val}")

# # Evaluate the models on the test set
# print("Test Set Evaluation:")

# # Linear Regression Evaluation
# lr_rmse_test = mean_squared_error(y_test, y_pred_lr_test, squared=False)

# # Decision Tree Evaluation
# dt_rmse_test = mean_squared_error(y_test, y_pred_dt_test, squared=False)

# # Random Forest Evaluation
# rf_rmse_test = mean_squared_error(y_test, y_pred_rf_test, squared=False)

# # Print out the test set evaluation results
# print(f"Linear Regression Test RMSE: {lr_rmse_test}")

# print(f"Decision Tree Test RMSE: {dt_rmse_test}")

# print(f"Random Forest Test RMSE: {rf_rmse_test}")



X = competition_df.drop(['id'], axis=1)

X = X.astype({col: 'category' for col in X.select_dtypes(include=['object']).columns})

competition_df['Price'] = lgb_model.predict(X)


Result = competition_df[['id', 'Price']]


Result.head()




