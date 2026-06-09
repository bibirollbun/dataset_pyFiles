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


!pip install --upgrade scikit-learn


!pip list


import sklearn
print(sklearn.__version__)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
ex_data = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


data


X_train = data.drop(columns= ['Price', 'id'])
X_train.head()


y_train = data['Price']
y_train


data.describe()


data.info()


# Making a pipeline

numerical_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand', 'Material', 'Style', 'Color']
ordinal_features = ['Size']
binary_features = ['Laptop Compartment', 'Waterproof']


print(data['Size'].unique())


size_categories = ['Small', 'Medium', 'Large']


from sklearn.preprocessing import FunctionTransformer


numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])


ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[size_categories]))
])


# def binary_encode(X):
#     print(type(X))
#     X_series = pd.Series(X.flatten())
#     X_series = X_series.str.lower().str.strip()
#     X_series = X_series.map({'Yes': 1, 'No': 0})
#     return X_series.to_numpy().reshape(-1, 1)

# def binary_encode(X):
#     print(X)
#     X = np.char.lower(X)
#     X = np.char.strip(X)

#     X = np.where(X == 'yes', 1, 0)
#     return X.reshape(-1, 1)

def binary_encode_pandas(X):
    return pd.DataFrame(X).applymap(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).to_numpy()


binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('binary', FunctionTransformer(binary_encode_pandas))
])


preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('ord', ordinal_pipeline, ordinal_features),
    ('binary', binary_pipeline, binary_features)
])


lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.01))
])


lasso_pipeline.fit(X_train, y_train)


lasso_pipeline.score(X_train, y_train)


y_pred_train_l = lasso_pipeline.predict(X_train)


rmse = root_mean_squared_error(y_train, y_pred_train_l)
rmse


X_ext = ex_data.drop(columns= ['Price', 'id'])
X_ext.head()


y_ext = ex_data['Price']
y_ext


X_combined = pd.concat([X_train, X_ext], ignore_index=True)
y_combined = pd.concat([y_train, y_ext], ignore_index=True)


lasso_pipeline.fit(X_combined, y_combined)


lasso_pipeline.score(X_combined, y_combined)


y_pred_l_ext = lasso_pipeline.predict(test)


test['Price'] = y_pred_l_ext
test[['id', 'Price']].to_csv('submission.csv', index=False)


from sklearn.ensemble import RandomForestRegressor


rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, ))
])


rf_pipeline.fit(X_combined, y_combined)


import xgboost
print(xgboost.__version__)





from xgboost import XGBRegressor


xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7))
])

# refer below documentation: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn


xgb_pipeline.fit(X_combined, y_combined)


xgb_pipeline.score(X_combined, y_combined)


y_pred_xgb = xgb_pipeline.predict(X_combined)


rmse_xbg = root_mean_squared_error(y_combined, y_pred_xgb)
rmse_xbg


y_pred_test_xgb = xgb_pipeline.predict(test)


test['Price'] = y_pred_test_xgb
test[['id','Price']].to_csv('submission.csv', index=False)




