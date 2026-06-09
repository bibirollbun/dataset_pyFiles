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


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train


#some feature engineering
train['was_contacted'] = (train['pdays'] != -1).astype(int) #a column to check if a specific customer was ever contacted or not
test['was_contacted'] = (test['pdays'] != -1).astype(int)


categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
binary_columns = ['default', 'housing', 'loan', 'was_contacted']
num_columns = []


#preprocessing categorical columns
from sklearn.preprocessing import OneHotEncoder

'''def preprocess_categories(df):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    transformed = encoder.fit_transform(df[categorical_columns])
    
    encoded_df = pd.DataFrame(transformed, 
                              columns=encoder.get_feature_names_out(categorical_columns),
                              index=df.index)
    
    df = df.drop(columns=categorical_columns)
    
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

train = preprocess_categories(train)'''


#preprocessing binary columns

'''def yes_or_no(binary_series):
    return binary_series.apply(lambda x: 1 if x == 'yes' else 0 if x == 'no' else x)

def preprocess_binary(df):
    for col in binary_columns:
        df[col] = yes_or_no(df[col])
    return df

train = preprocess_binary(train)'''


#join the two preprocessing funcs

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(train[categorical_columns])  # Fit on train only

def preprocess_categories(df):
    transformed = encoder.transform(df[categorical_columns])
    encoded_df = pd.DataFrame(
        transformed,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )
    df[binary_columns] = df[binary_columns].replace({'yes': 1, 'no': 0})
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)
    return df

# now use it on both
train = preprocess_categories(train)
test = preprocess_categories(test)


train['balance_per_campaign'] = train['balance'] / (train['campaign'] + 1)
test['balance_per_campaign'] = test['balance'] / (test['campaign'] + 1)


import lightgbm as lgb
from xgboost import XGBClassifier

model2 = XGBClassifier(use_label_encoder=False,
                       eval_metric='logloss',
                       n_estimators=100,
                       max_depth=2,
                       learning_rate=1)


model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42
)
x_train = train.drop(columns = ['id', 'y'])
y_train = train['y']


model.fit(x_train, y_train)
model2.fit(x_train, y_train)


test


x_test = test.drop(columns = ['id'])
y_pred = (model.predict_proba(x_test)[:,1] + model2.predict_proba(x_test)[:,1]) / 2


submission = pd.DataFrame({
    'id' : test['id'],
    'y' : y_pred
})


submission.to_csv('submission.csv', index = False)




