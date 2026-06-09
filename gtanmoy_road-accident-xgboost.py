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
from scipy import stats


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print(f" Train shape {df_train.shape}")
print(f" Test shape {df_test.shape}")
print(f" Sample shape {df_submission.shape}")


df_train.head(5)


df_test.head(5)


df_train = df_train.drop(columns='id')
df_test = df_test.drop(columns='id')


df_train.head()


df_test.head()


df_train.info()


df_train.describe()


df_train.isnull().sum().sum()


df_test.isnull().sum().sum()


# Duplicated Rows
df_train.duplicated().sum()


df_test.duplicated().sum()


#Drop Duplicate Rows
df_train = df_train.drop_duplicates()
df_train.duplicated().sum()


df_num_cols = df_train.select_dtypes(include=['float64', 'int64'])
df_num_cols


df_bool_cols = df_train.select_dtypes(include=['bool'])
df_bool_cols


df_obj_cols = df_train.select_dtypes(include=['object'])
df_obj_cols


con_num_cols = ['curvature', 'accident_risk']
dis_num_cols = ['num_lanes', 'speed_limit',	'num_reported_accidents']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
obj_cols = ['road_type', 'lighting', 'weather', 'time_of_day']


df_train.hist(bins=20, figsize=(10,8))
plt.show()


plt.scatter(df_train['curvature'], df_train['accident_risk'])
plt.xlabel('curvature')
plt.ylabel('accident_risk')
plt.legend()
plt.show()


for col in con_num_cols:
    plt.hist(df_train[col], bins =25, density=True, label=col)
    plt.xlabel(col)
    plt.ylabel(f'P({col})')
    plt.legend()
    plt.show()


stats.probplot(df_train['accident_risk'], dist="norm", plot=plt)
stats.probplot(np.log1p(df_train['accident_risk']), dist="norm", plot=plt)
plt.show()


for col in dis_num_cols:
    print(col, df_train[col].nunique(), df_train[col].unique())


for col in dis_num_cols:
    plt.figure(figsize=(5, 3))
    median_prices = df_train.groupby(col)['accident_risk'].mean()
    median_prices.plot.bar()
    plt.title(f'mean accident_risk by {col}')
    plt.ylabel('mean accident_risk')
    plt.xlabel(col)
    plt.show()




for col in bool_cols:
    
    mean_risk = df_train.groupby(col)['accident_risk'].mean()
    plt.figure(figsize=(5, 3))
    plt.bar(mean_risk.index.astype(str), mean_risk.values)  # Convert boolean to string for x-axis labels
    plt.title(f'Mean accident_risk by {col}')
    plt.xlabel(col)
    plt.ylabel('Mean accident_risk')
    plt.show()


for col in obj_cols:
    print(col, df_train[col].nunique(), df_train[col].unique())


for col in obj_cols:
    plt.figure(figsize=(5, 3))
    data = df_train.copy()
    median_prices = data.groupby(col)['accident_risk'].median()
    median_prices.plot.bar()
    plt.title(f'Median accident_riskby {col}')
    plt.xticks(rotation=0)
    plt.show()



for col in obj_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=col, y='accident_risk', data=df_train)
    plt.title(f'accident_risk Distribution by {col}')
    plt.xticks(rotation=45)
    plt.show()



correlation_matrix = df_train.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True,  cmap='coolwarm', fmt=".3f")



df_train = df_train.copy()

df_train['speed_squared'] = df_train['speed_limit'] ** 2
df_train['curvature_squared'] = df_train['curvature'] ** 2
df_train['is_night'] = (df_train['lighting'] == 'night').astype(int)
df_train['bad_weather'] = (df_train['weather']  != 'clear').astype(int)
df_train['speed_curvature'] = df_train['speed_limit'] * df_train['curvature']
df_train['meta_speed'] = (df_train['speed_limit'] >= 60).astype(int)
df_train['dangerous'] = (
    (df_train['curvature'] > 0.5).astype(int) +
    (df_train['speed_limit'] >= 60).astype(int) +
    df_train['bad_weather'] +
    df_train['is_night'] +
    (df_train['num_reported_accidents'] >= 2).astype(int)
)


df_train


df_test = df_test.copy()

df_test['speed_squared'] = df_test['speed_limit'] ** 2
df_test['curvature_squared'] = df_test['curvature'] ** 2
df_test['is_night'] = (df_test['lighting'] == 'night').astype(int)
df_test['bad_weather'] = (df_test['weather']  != 'clear').astype(int)
df_test['speed_curvature'] = df_test['speed_limit'] * df_test['curvature']
df_test['meta_speed'] = (df_test['speed_limit'] >= 60).astype(int)
df_test['dangerous'] = (
    (df_test['curvature'] > 0.5).astype(int) +
    (df_test['speed_limit'] >= 60).astype(int) +
    df_test['bad_weather'] +
    df_test['is_night'] +
    (df_test['num_reported_accidents'] >= 2).astype(int)
)


df_test


obj_cols, bool_cols


columns=['road_type', 'lighting', 'weather', 'time_of_day']
df_train_encoded = pd.get_dummies(df_train, columns = obj_cols, drop_first = True)
df_test_encoded = pd.get_dummies(df_test, columns = obj_cols, drop_first = True)


df_train_encoded


df_test_encoded


df_test_encoded.isna().sum().sum()


for col in df_train_encoded.select_dtypes(include='bool').columns:
    df_train_encoded[col] = df_train_encoded[col].astype(int)
    df_test_encoded[col] = df_test_encoded[col].astype(int)


df_train_encoded


df_test_encoded


df_test_encoded.isna().sum().sum()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb



X_train = df_train_encoded.drop('accident_risk', axis=1)
y_train = df_train_encoded['accident_risk']

X_test = df_test_encoded.copy()


xgb = XGBRegressor(random_state=123, verbosity=0)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',  # Use negative MSE for scoring
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)

# Convert the negative MSE back to positive for interpretability
best_neg_mse = grid_search.best_score_
best_mse = -best_neg_mse
print("Best CV MSE:", best_mse)
print("Best CV RMSE:", np.sqrt(best_mse))

# Evaluate best model on test set
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)


df_test.shape, y_pred.shape


submission = pd.DataFrame({'id': df_submission['id'], 'accident_risk': y_pred})


submission.describe()


submission.to_csv('submission.csv', index=False)




