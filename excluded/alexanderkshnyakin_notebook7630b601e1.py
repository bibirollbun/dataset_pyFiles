import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import sqrt


data = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')
data_test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv.zip')
submission = pd.read_csv('../input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')


data.head()


data.info()


target_col = 'y'



data.isnull().sum()


data.isna().sum()


for i in data.isna().sum():
  if i != 0:
    print(i)


data.describe()


y = data['y']
X = data.drop(['y'], axis = 1)


X.info()


numerical_features = X.select_dtypes(include = 'number').columns.values
categorical_features = X.select_dtypes(exclude = 'number').columns.values


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 35)


encoding = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)


encoding.fit(X_train[categorical_features])

X_train[categorical_features] = encoding.transform(X_train[categorical_features])
data_test[categorical_features] = encoding.transform(data_test[categorical_features])
X_val[categorical_features] = encoding.transform(X_val[categorical_features])


imp = SimpleImputer(strategy = 'median')


imp.fit(X_train)

X_train = imp.transform(X_train)
data_test = imp.transform(data_test)
X_val = imp.transform(X_val)



selector = SelectKBest(mutual_info_regression, k = 40)


selector.fit(X_train, y_train)

X_train = selector.transform(X_train)

data_test = selector.transform(data_test)

X_val = selector.transform(X_val)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=5,n_estimators=100, max_features=0.7, criterion='friedman_mse')
lr = LinearRegression()


rf.fit(X_train, y_train)
lr.fit(X_train, y_train)


y_train_pred1 = rf.predict(X_train)
y_val_pred1 = rf.predict(X_val)

y_train_pred2 = lr.predict(X_train)
y_val_pred2 = lr.predict(X_val)


print(r2_score(y_train, y_train_pred1))
print(r2_score(y_val, y_val_pred1))

print(r2_score(y_train, y_train_pred2))
print(r2_score(y_val, y_val_pred2))


y_test_pred = rf.predict(data_test)


submission['y'] = y_test_pred[:]



submission.to_csv('rf_submission.csv', index= False)


print(sqrt(mean_squared_error(y_val, y_val_pred1)))

