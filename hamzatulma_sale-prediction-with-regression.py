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


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df.head()


df.isnull().sum()


test.isnull().sum()


mod_values= df['num_sold'].mode().iloc[0]
df = df.fillna(mod_values)
df.isnull().sum()


from sklearn.preprocessing import LabelEncoder

# LabelEncoder nesnesi oluşturma
le = LabelEncoder()

# Kategorik sütunları seçme
categorical_columns = ['country', 'store', 'product']

# Her bir kategorik sütunu dönüştürme
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    test[col] = le.fit_transform(test[col])


df.head()


def fix_dates(df, date_column='date'):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    
    df.drop(columns=[date_column], inplace=True)
    
    return df

df = fix_dates(df, date_column='date')
test= fix_dates(test, date_column='date')


df.head()


test.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df.drop(columns=['num_sold'], axis=1)
y = df['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)


model.fit(X, y)

prediction = model.predict(test)


prediction


sub['num_sold']=prediction
sub.to_csv('tahminlerim.csv', index=False)  # save !


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
newpred = model.predict(test)


sub['num_sold']=newpred
sub.to_csv('tahmi2.csv', index=False)  # save !




