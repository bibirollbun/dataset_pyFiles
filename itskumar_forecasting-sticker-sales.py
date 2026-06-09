# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dataset = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
dataset.head()


dataset.shape


dataset.describe()


dataset.select_dtypes(['object']).describe()


some_data = dataset.iloc[:1000, :]


%matplotlib inline

dataset['num_sold'].hist();


dataset.groupby('country')['num_sold'].sum().plot(kind='bar');


dataset.groupby('store')['num_sold'].sum().plot(kind='bar');


dataset.groupby('product')['num_sold'].sum().plot(kind='bar');


dataset['date'] = pd.to_datetime(dataset.date)


dataset.groupby(dataset['date'].dt.year)['num_sold'].sum().plot(kind='bar')
plt.xlabel('year')
plt.ylabel('Total Sales');


dataset['month'] = dataset['date'].dt.month
dataset['year'] = dataset['date'].dt.year
dataset['quarter'] = dataset['date'].dt.quarter

dataset.groupby(['year', 'month'])['num_sold'].sum().reset_index()


dataset['day_of_week'] = dataset['date'].dt.dayofweek
dataset['is_weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)


dataset.head()


corr_matrix = dataset.select_dtypes(['number', 'datetime']).corr()
corr_matrix['num_sold'].sort_values(ascending=False)


num_features = ['is_weekend', 'day_of_week', 'month', 'year', 'quarter']
cat_features = ['country', 'store', 'product']


dataset.isnull().sum()


# dataset['num_sold'] = dataset['num_sold'].fillna(dataset['num_sold'].mean())
dataset = dataset.dropna(subset=['num_sold'])


dataset.boxplot(['num_sold']);


# Detecting outliers using IQR 
Q1 = dataset['num_sold'].quantile(0.25)
Q3 = dataset['num_sold'].quantile(0.75) 
IQR = Q3 - Q1


data_cleaned = dataset[(dataset['num_sold'] >= (Q1 - 1.5 * IQR)) & (dataset['num_sold'] <= (Q3 + 1.5 * IQR))]


data_cleaned.boxplot(['num_sold']);


data_cleaned.shape


X = data_cleaned[num_features + cat_features]
y = data_cleaned['num_sold']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    # ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor= ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])


X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)


y_pred = lin_reg.predict(X_test_prepared)


from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(y_test, y_pred)


# from sklearn.svm import SVR

# regressor = SVR(kernel='rbf')
# regressor.fit(X_train_prepared, y_train)


# y_pred = regressor.predict(X_test_prepared)
# mean_absolute_percentage_error(y_test, y_pred)


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(X_train_prepared, y_train)


y_pred = dt.predict(X_test_prepared)
mean_absolute_percentage_error(y_test, y_pred)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train_prepared, y_train)


y_pred = rf.predict(X_test_prepared)
mean_absolute_percentage_error(y_test, y_pred)


from xgboost import XGBRegressor

xgbr = XGBRegressor(objective ='reg:squarederror', n_estimators=100)
xgbr.fit(X_train_prepared, y_train)


y_pred = xgbr.predict(X_test_prepared)
mean_absolute_percentage_error(y_test, y_pred)


print("Training Score:")
y_pred = rf.predict(X_train_prepared)
mape = mean_absolute_percentage_error(y_train, y_pred)
print(mape)

print("Testing Score:")
y_pred = rf.predict(X_test_prepared)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(mape)


# from sklearn.model_selection import GridSearchCV

# param_grid = { 'n_estimators': [100, 200, 500], 
#               'max_depth': [None, 10, 20], 
#               'min_samples_split': [2, 5, 10], 
#             } 

# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=param_grid, 
#                            cv=3, n_jobs=-1, 
#                            scoring='neg_mean_absolute_percentage_error') 
# grid_search.fit(X_train_prepared, y_train)


# best_params = grid_search.best_params_ 
# best_model = grid_search.best_estimator_
# best_params


# print("Best Model:")
# print("Training Score:")
# y_pred = best_model.predict(X_train_prepared)
# mape = mean_absolute_percentage_error(y_train, y_pred)
# print(mape)

# print("Testing Score:")
# y_pred = best_model.predict(X_test_prepared)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# print(mape)


test_set = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test_set.head()


test_set.shape


test_set['date'] = pd.to_datetime(test_set['date'])
test_set['month'] = test_set['date'].dt.month
test_set['year'] = test_set['date'].dt.year
test_set['quarter'] = test_set['date'].dt.quarter
test_set['day_of_week'] = test_set['date'].dt.dayofweek
test_set['is_weekend'] = test_set['day_of_week'].isin([5, 6]).astype(int)


test_set.head()


test_set_prepared = preprocessor.transform(test_set[num_features+cat_features])


predictions = dt.predict(test_set_prepared)


submission = pd.DataFrame({'id': test_set['id'],
                          'num_sold': predictions})
submission.tail()


submission.shape


submission.to_csv('submission.csv', index=False)




