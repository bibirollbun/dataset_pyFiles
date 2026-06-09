# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import metrics

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv', index_col=0)
train_df.head(10)


# drop unnecessary columns
train_df.drop(['airline', 'flight', 'source_city', 'destination_city'], axis=1, inplace=True)


X = train_df.drop('price', axis=1)
y = train_df['price']


X.info()


cat_attributes = X.select_dtypes(include='object').columns.to_list()
num_attributes = X.select_dtypes(include=['float64','int64']).columns.to_list()

print('categories:', cat_attributes)
print('numerical:', num_attributes)


cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('numeric', num_pipeline, num_attributes),
    ('categorical', cat_pipeline, cat_attributes)
])


X_prepared = preprocessor.fit_transform(X)


# split to train and test
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


coefficients = lr_model.coef_
theta0 = lr_model.intercept_

print('coefficients:', coefficients)
print('theta0:', theta0)


y_pred = lr_model.predict(X_test)


MAE = metrics.mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(f"MAE: {MAE:.2f}")
print(f"RMSE: {RMSE:.2f}")



# MLPR
mlpr_model = MLPRegressor(max_iter=2000)
mlpr_model.fit(X_train, y_train)


mlpr_model_pred = mlpr_model.predict(X_test)

MAE = metrics.mean_absolute_error(y_test, mlpr_model_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, mlpr_model_pred))

print(f"MAE: {MAE:.2f}")
print(f"RMSE: {RMSE:.2f}")


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(max_depth=10)
rf_model.fit(X_train, y_train)
rf_model_pred = rf_model.predict(X_test)

MAE = metrics.mean_absolute_error(y_test, rf_model_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, rf_model_pred))

print(f"MAE: {MAE:.2f}")
print(f"RMSE: {RMSE:.2f}")


# submission
test_df.head()


test_df.drop(['airline', 'flight', 'source_city', 'destination_city'], axis=1, inplace=True)


test_df_prepared = preprocessor.transform(test_df)


y_test_predicted = rf_model.predict(test_df_prepared)


submission = pd.DataFrame({
    'id':test_df.index,
    'price':y_test_predicted
})


submission.to_csv('submission.csv', index=False)




