# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train2 = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


train2.head()


test.head()


train.shape, train2.shape


plt.hist(x=train['Price'])
plt.title("distribution of price")



train.isnull().sum()


train.describe()


import seaborn as sns

sns.pairplot(train, diag_kind='kde')
plt.show();


numeric_data = train.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
corr_matrix['Price'].sort_values(ascending=False)


train_merged = pd.concat([train, train2], ignore_index=False)


train_merged.shape


train_merged.isnull().sum()


cols_with_na = [c for c in train_merged.columns if train_merged[c].isna().sum()>0]
cols_with_na


sns.boxplot(x='Weight Capacity (kg)', y='Brand', data=train2)
plt.title('Average Weight based on brand');


train_merged.dtypes


categorical_cols = [col for col in train_merged.columns if train_merged[col].dtype == 'O']
numerical_cols = [col for col in train_merged.columns if train_merged[col].dtype != 'O']
numerical_cols.remove('Price')

print(categorical_cols)
print(numerical_cols)


for col in categorical_cols:
    print(col, ":", train_merged[col].unique())


from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

train_merged[numerical_cols] = numerical_imputer.fit_transform(train_merged[numerical_cols])
train_merged[categorical_cols] = categorical_imputer.fit_transform(train_merged[categorical_cols])


train_merged.isnull().sum()


test[numerical_cols] = numerical_imputer.transform(test[numerical_cols])
test[categorical_cols] = categorical_imputer.transform(test[categorical_cols])


# feature engineering
train_merged['Weight per Compartment'] = train_merged['Weight Capacity (kg)'] / train_merged['Compartments']

test['Weight per Compartment'] = test['Weight Capacity (kg)'] / test['Compartments']
# train_merged.head()


size_mappings = {'Small': 0, 'Medium': 1, 'Large': 2}
train_merged['Size'] = train_merged['Size'].map(size_mappings)


test['Size'] = test['Size'].map(size_mappings)

train_merged['Laptop Compartment'] = train_merged['Laptop Compartment'].map({'Yes': 1, 'No': 0})
train_merged['Waterproof'] = train_merged['Waterproof'].map({'Yes': 1, 'No': 0})

test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})
test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})


train_merged.head()


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')
ohe_cols = ['Brand', 'Material', 'Style', 'Color']
encoded_cols = ohe.fit_transform(train_merged[ohe_cols])
encoded_cols_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(ohe_cols))
train_merged = train_merged.drop(columns=ohe_cols).join(encoded_cols_df)

# train_merged.head()



encoded_cols = ohe.transform(test[ohe_cols])
encoded_cols_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(ohe_cols))
test = test.drop(columns=ohe_cols).join(encoded_cols_df)


from sklearn.model_selection import train_test_split
X= train_merged.drop(columns=['id', 'Price'])
y = train_merged['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

# test = sc.transform(test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('MSE:', mean_squared_error(y_val, y_pred, squared=False))


from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(criterion='squared_error', max_depth=5,)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('MSE:', mean_squared_error(y_val, y_pred, squared=False))



# from sklearn.ensemble import RandomForestRegressor

# rf_model = RandomForestRegressor(n_estimators=100)
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_val)

# print("MAE:", mean_absolute_error(y_val, y_pred))
# print('MSE:', mean_squared_error(y_val, y_pred, squared=False))



from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100,
                         max_depth=5,
                         gamma=0.1,
                         min_child_weight=3,
                        eval_metric='rmse')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('MSE:', mean_squared_error(y_val, y_pred, squared=False))



from lightgbm import LGBMRegressor

lgb_model = LGBMRegressor(max_depth=-1, num_leaves=31, n_estimators=100)
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, y_pred))
print('MSE:', mean_squared_error(y_val, y_pred, squared=False))


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'num_leaves': [31, 50, 100],
#     'max_depth': [-1, 3, 5, 7, 10],
#     # 'min_data_in_leaf': [20, 50, 100],
#     # 'feature_fraction': [0.6, 0.8, 1.0],
#     # 'bagging_fraction': [0.6, 0.8, 1.0],
#     # 'bagging_freq': [1, 5, 10],
#     # 'min_split_gain': [0, 0.1, 0.2]
# }

# grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best parameters:", best_params)


# Train the LightGBM model with the best parameters
# best_lgbm_model = LGBMRegressor(**best_params, random_state=42)
# best_lgbm_model.fit(X_train, y_train)
# y_pred = best_lgbm_model.predict(X_val) 

# # Evaluate the model
# print("Optimized MAE:", mean_absolute_error(y_val, y_pred))
# print("Optimized RMSE:", mean_squared_error(y_val, y_pred, squared=False)) 


test_scaled = sc.transform(test.drop('id', axis=1))
# predictions = xgb_model.predict(test_scaled)
predictions = lgb_model.predict(test_scaled)


submission = pd.DataFrame({'id': test['id'], 'Price': predictions})


submission.head()


submission.to_csv('submission.csv', index=False)


submission.shape




