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


import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
X_test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train = train_df.copy()
X_test = X_test_df.copy()
y_test = sub.copy()


# Remove Unique columns like 'id'

train.drop('id', axis = 1, inplace = True)
X_test.drop('id', axis = 1, inplace = True)
y_test.drop('id', axis = 1, inplace = True)


train.head()


train.info()


train.describe()


train.drop_duplicates(inplace = True)
len(train)


# encode categorical columns.
from sklearn.preprocessing import LabelEncoder, StandardScaler

encode = LabelEncoder()
train['Sex'] = encode.fit_transform(train['Sex'])
X_test['Sex'] = encode.transform(X_test['Sex'])


sns.histplot(train['Calories'])


sns.heatmap(train.corr(), annot = True, cmap = 'coolwarm')


# we can see height has lowest correlation with target label so drop 'height' column from both train and test data.
train.drop('Height', axis = 1, inplace = True)
X_test.drop('Height', axis = 1, inplace = True)


# Now remove outliers.

Q1 = train['Calories'].quantile(0.25)
Q3 = train['Calories'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

train = train[(train['Calories'] > lower_bound) & (train['Calories'] < upper_bound)]
len(train)


# spliting train data.

X_train = train.drop('Calories', axis = 1)
y_train = train['Calories']


# Standardization.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.metrics import mean_squared_error

def model_eval(model, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    print(RMSE)


# lin_reg = LinearRegression()
# model_eval(lin_reg)
print('61.366589131461446')


# lasso = Lasso()
# model_eval(lasso)
print('59.918691531768346')


# ridge = Ridge()
# model_eval(ridge)
print('61.36650587456538')


# dtree = DecisionTreeRegressor(max_depth = 3)
# model_eval(dtree)
print('60.139196667138826')


# rf = RandomForestRegressor(n_estimators = 100, max_depth = 4, random_state = 101)
# model_eval(rf)
print('60.74920287410416')


# bagg = BaggingRegressor(estimator = dtree, n_jobs = 3, verbose = 2)
# model_eval(bagg)
print('60.154611802025215')


# Gradient Boosting


gbr = GradientBoostingRegressor()
# model_eval(gbr)
print('60.154611802025215')


# param_grid = {
#     'n_estimators' : [100],
#     'learning_rate' : [0.01,0.1],
#     'max_depth' : [3, 5],
#     'min_samples_split' : [2, 5]
# }

# grid_search = GridSearchCV(estimator = gbr, param_grid = param_grid, n_jobs = -1, cv = 5, verbose = 2, scoring = 'neg_root_mean_squared_error')

# grid_search.fit(X_train, y_train)

# print('RMSE :', -grid_search.best_score_)
print('3.867758040489283')


# xgb = XGBRegressor(random_state = 101)

# param_grid = {
#     'n_estimators' : [100],
#     'max_depth' : [3,5],
#     'learning_rate' : [0.01,0.1],
#     'subsample' : [0.8, 1.0],
#     'colsample_bytree' : [0.8,1.0]
# }

# grid_search = GridSearchCV(
#     estimator = xgb, 
#     param_grid = param_grid, 
#     cv = 5,
#     scoring = 'neg_root_mean_squared_error',
#     n_jobs = -1,
#     verbose = 1
# )

# grid_search.fit(X_train, y_train)
# print('best parameters :', grid_search.best_params_)
# print('RMSE :', -grid_search.best_score_)


best_param = {
    'n_estimators' : 100,
    'max_depth' : 5,
    'learning_rate' : 0.1,
    'subsample' : 0.8,
    'colsample_bytree' : 0.8
}


final_model = XGBRegressor(**best_param)
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)


submission = pd.DataFrame({
    'id': X_test_df['id'],
    'Calories': final_pred
})
submission.to_csv('submission.csv', index=False)

