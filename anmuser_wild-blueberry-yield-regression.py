import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


train = pd.read_csv('/kaggle/input/playground-series-s3e14/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e14/test.csv')
sam = pd.read_csv('/kaggle/input/playground-series-s3e14/sample_submission.csv')


train.isna().sum()


train.head(3)


X = train.drop(columns=['yield','id'])
y = train['yield']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=1000,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared (R² Score): {r2}')


test1 = test.drop(columns=['id'])


Predictions = model.predict(test1)


sam['yield']= Predictions
sam['id'] = test['id']
sam.to_csv('submission.csv',index=False)


sam.head(3)

