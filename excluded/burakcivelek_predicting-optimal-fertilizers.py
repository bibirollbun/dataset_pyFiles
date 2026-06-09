# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
print(train_df.isnull().sum())
print(test_df.isnull().sum())
print(train_df.info())
train_df = train_df.drop('id', axis=1)


train_df.head()


cat_cols = list(train_df.select_dtypes(include=['object']).columns)
le = LabelEncoder()
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    if col != 'Fertilizer Name':
        test_df[col] = le.fit_transform(test_df[col])
    


train_df.head()


y_train = train_df['Fertilizer Name']
x_train = train_df.drop('Fertilizer Name', axis=1)


xgb = XGBClassifier(n_estimators=1000, random_state=61)
model = xgb.fit(x_train, y_train)


x_test = test_df.drop('id', axis=1)


y_pred = model.predict(x_test)
y_pred = le.inverse_transform(y_pred)



sub = pd.DataFrame()


sub['id'] = test_df['id']
sub['Fertilizer Name'] = y_pred


sub.to_csv('/kaggle/working/submission.csv', index=False)

