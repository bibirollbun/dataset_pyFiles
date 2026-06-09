from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/logical-rhythm-2k21-cars24/train.csv')
df.head()


X = df[['model_year', 'maker', 'model_name', 'city', 'distance_covered (km)', 'pre_owner']]
X.head()


maker_le = LabelEncoder()
X['maker_encoded'] = maker_le.fit_transform(X['maker'])
X = X.drop('maker', axis=1)
model_name_le = LabelEncoder()
X['model_name_encoded'] = model_name_le.fit_transform(X['model_name'])
X = X.drop('model_name', axis=1)
city_le = LabelEncoder()
X['city_encoded'] = city_le.fit_transform(X['city'])
X = X.drop('city', axis=1)
pre_owner_le = LabelEncoder()
X['pre_owner_encoded'] = pre_owner_le.fit_transform(X['pre_owner'])
X = X.drop('pre_owner', axis=1)
X.head()


y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(random_state=42)


param_grid = {'n_estimators': [50, 100]}
grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_log_error', cv=5)
grid.fit(X_train, y_train)


y_pred = grid.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("RMSLE:", rmsle)


df = pd.read_csv('/kaggle/input/logical-rhythm-2k21-cars24/test.csv')
final = df[['car ID']]
df = df[['model_year', 'maker', 'model_name', 'city', 'distance_covered (km)', 'pre_owner']]
df['maker_encoded'] = maker_le.transform(df['maker'])
df = df.drop('maker', axis=1)
df['model_name_encoded'] = model_name_le.transform(df['model_name'])
df = df.drop('model_name', axis=1)
df['city_encoded'] = city_le.transform(df['city'])
df = df.drop('city', axis=1)
df['pre_owner_encoded'] = pre_owner_le.transform(df['pre_owner'])
df = df.drop('pre_owner', axis=1)
final['price'] = grid.predict(df)
final.to_csv('submission.csv', index=False)

