import torch
import torch.nn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.dtypes


obj_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in obj_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.fit_transform(test_df[col])


train_df


y = train_df['y']
train_df = train_df.drop('y', axis=1)
X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2)


X_train.head()


y_train.head()


xgbc = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.05, objective='binary:logistic')
xgbc.fit(X_train, y_train)
preds = xgbc.predict(X_val)


acc = accuracy_score(y_val, preds)
print(acc)


sub_pred = xgbc.predict(test_df)
pred_df = pd.DataFrame({'id': test_df['id'], 'y': sub_pred})
pred_df.head()


pred_df.to_csv('submission.csv', index=False)

