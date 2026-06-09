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


# Loading Dataset
train = pd.read_csv('../input/computer-prices-2025/computer_prices_all.csv')
test = pd.read_csv('../input/computer-prices-2025/computer_prices_test.csv')
train = train.drop('ID', axis=1)
y = train.pop('price')

train.head()


train.nunique()


test.nunique()


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor



# Ordinal Encoding for High Cardinality Column
ord_col = ['model', 'cpu_model', 'gpu_model', 'weight_kg']

ord = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)

ord_train = pd.DataFrame(ord.fit_transform(train[ord_col]))
ord_train.columns = ord_col

ord_test = pd.DataFrame(ord.transform(test[ord_col]))
ord_test.columns = ord_col
print("Ordinal Encoding Done")

# OneHotEncoding for remaining Features
oh_cols = train.drop(ord_col, axis =1).columns
oh = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

train_set = pd.DataFrame(oh.fit_transform(train[oh_cols]))
train_set.columns = train_set.columns.astype('str')

test_set = pd.DataFrame(oh.transform(test[oh_cols]))
test_set.columns = test_set.columns.astype('str')
print("One Hot Encoding Done")

# Final train and test set
train_set = train_set.join(ord_train)
test_set = test_set.join(ord_test)

print("Dataset Ready")


model = XGBRegressor(objective='reg:squarederror', n_estimators = 5000, 
                     learning_rate = 0.005, enable_categorical = True, 
                     random_state = 0, early_stopping_rounds=100)


kf = KFold(n_splits = 5, shuffle = False)


oof=pd.Series(index=train_set.index)


scores=[]
i=0
for train_index, val_index in kf.split(train_set):
    i+=1
    print(f'fold{i} starting:')
    train, y_all = train_set.iloc[train_index], y.iloc[train_index]
    X_test, y_test = train_set.iloc[val_index], y.iloc[val_index]

    X_train, X_val, y_train, y_val = train_test_split(train, y_all, test_size=0.2, random_state=6)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)
    pred = model.predict(X_test)
    oof.iloc[val_index]=pred
    score = root_mean_squared_error(y_test, pred)
    scores.append(score)
    print(f'Training completed! RMSE score : {score}')
print("Fully completed!!")
cv_score = 0
for i in scores:
    cv_score+=i
cv_score/=5
print(f"CV SCORE : {cv_score}")


X_train, X_val, y_train, y_val = train_test_split(train_set, y, test_size=0.2, random_state=6)


model.fit(train_set, y, eval_set=[(X_val, y_val)], verbose=1000)


pred = model.predict(test_set)

sub = pd.DataFrame({'ID':test.ID, 'price':pred})
oof.to_csv('oof', index=False)
print("Viewing Submission")
print(sub.head())

sub.to_csv('submission.csv', index = False)
print("\nSubmitted")

