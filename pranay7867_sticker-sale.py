import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

train.head()


test.head()


print(train.isnull().sum())


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

train['num_sold'].fillna(train['num_sold'].median(), inplace=True)

def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

train = create_date_features(train)
test = create_date_features(test)

train = pd.get_dummies(train, columns=['country', 'store', 'product'], drop_first=True)
test = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=True)

test = test.reindex(columns=train.columns.drop("num_sold"), fill_value=0)

X = train.drop(columns=['id', 'date', 'num_sold'])
y = train['num_sold']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
print(f'Mean Absolute Error: {mae}')

test_preds = model.predict(test.drop(columns=['id', 'date']))
test['num_sold'] = test_preds

test[['id', 'num_sold']].to_csv("submission.csv", index=False)





