import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.head()


test_df


train_df.isnull().sum()


test_df.isnull().sum()


train_df.dtypes


obj_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 'public_road', 'holiday', 'school_season']

for col in obj_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


train_df.head()


test_df.head()


y = train_df['accident_risk']
train_df = train_df.drop(['accident_risk', 'id'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2)


xgbr = XGBRegressor()
xgbr.fit(X_train, y_train)
pred = xgbr.predict(X_test)


mse = mean_squared_error(y_test, pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")


test_ids = test_df['id']
test_df = test_df.drop('id', axis=1)


pred = xgbr.predict(test_df)


submission = pd.DataFrame({'id': test_ids, 
              'accident_risk': pred})


submission


submission.to_csv('/kaggle/working/submission.csv', index=False)

