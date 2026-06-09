# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


train_df.info()


test_df.info()


train_df.isnull().sum()


train_df = train_df.drop(columns=['id','Name'])
test_df = test_df.drop(columns=['Name'])


num_cols = [i for i in train_df.select_dtypes(include=['float64','int64']).columns if i in test_df.columns]


train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].mean())
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].mean())


train_df['Profession'] = train_df['Profession'].fillna(train_df['Profession'].mode()[0])
test_df['Profession'] = test_df['Profession'].fillna(test_df['Profession'].mode()[0])


train_df.isnull().sum()


test_df.isnull().sum()


train_df = train_df.dropna(axis=0)
test_df = test_df.dropna(axis=0)


train_df.head(3)


for_dummies = [i for i in list(train_df.select_dtypes(include=['object']).columns)]


train_df = pd.get_dummies(train_df, columns=for_dummies, drop_first=True)
test_df = pd.get_dummies(test_df, columns=for_dummies, drop_first=True)


train_df.shape


test_df.shape


for i in train_df:
    if not i in test_df.columns:
        if i == 'Depression':
            continue
        else:
            train_df = train_df.drop(f'{i}', axis=1)

for i in test_df:
    if not i in train_df.columns:
        if i == 'id':
            continue
        else:
            test_df = test_df.drop(f'{i}', axis=1)


train_df.shape


test_df.shape


train_df.head()


test_df.head()


y_train = train_df['Depression']
x_train = train_df.drop('Depression', axis=1)


x_test = test_df.drop('id',axis=1)


rf = RandomForestClassifier(n_estimators=200)
model = rf.fit(x_train, y_train)


y_pred = model.predict(x_test)


sub = pd.DataFrame()


sub['id'] = test_df['id']


sub['Depression'] = y_pred


sub.to_csv('/kaggle/working/submission.csv', index=False)




