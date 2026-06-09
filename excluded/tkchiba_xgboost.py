import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


#日付の変換
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
df['quarter'] = df['date'].dt.quarter

df_test['date'] = pd.to_datetime(df_test['date'])

df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['dayofyear'] = df_test['date'].dt.dayofyear
df_test['quarter'] = df_test['date'].dt.quarter


df.dropna(inplace=True)
y_col = 'num_sold'
y = df[y_col]
X = df.drop(labels=[y_col, 'id', 'date'], axis=1)
X_test = df_test.drop(labels=['id', 'date'], axis=1)

# Label Encoding
oe = OrdinalEncoder()
oe.set_output(transform='pandas')
cat_cols = X.select_dtypes(exclude=np.number).columns.to_list()
X[cat_cols] = oe.fit_transform(X[cat_cols])
X_test[cat_cols] = oe.transform(X_test[cat_cols])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=0)


xgb = XGBRegressor(learning_rate=0.01,
              eval_metric='rmse',
              early_stopping_rounds=10,
              importance_type='total_gain',
              random_state=0)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)


y_pred = xgb.predict(X_test)


sample_submission

sub = sample_submission
sub['num_sold'] = list(map(int, y_pred))
sub.to_csv("submission.csv", index=False)


sub




