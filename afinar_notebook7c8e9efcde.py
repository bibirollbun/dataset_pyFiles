# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv(dirname + '/train.csv')
test = pd.read_csv(dirname + '/test.csv')
submission = pd.read_csv(dirname + '/sample_submission.csv')


display(
    train.head(100), train.shape,
    test.head(), test.shape
)


train.info()


size_of_null_num_sold = train['num_sold'].isnull().sum()

plt.pie(x=[len(train['num_sold']), size_of_null_num_sold],
        labels=['notNULL', 'NULL'],
        autopct='%.2f%%'
       )
plt.title('Proportion of num_sold')
plt.show()


display(
    len(train['date'].unique()),
    train['country'].unique(),
    train['store'].unique(),
    train['product'].unique()
)


train = train.dropna(subset=['num_sold'])


import holidays

holiday_dict = {
    'Canada': holidays.Canada(),
    'Finland': holidays.Finland(),
    'Italy': holidays.Italy(),
    'Kenya': holidays.Kenya(),
    'Norway': holidays.Norway(),
    'Singapore': holidays.Singapore()
}

# 공휴일 여부 컬럼 추가
def is_holiday(row):
    country = row['country']
    date = row['date']
    if country in holiday_dict:
        return date in holiday_dict[country]
    return False

train['is_holiday'] = train.apply(is_holiday, axis=1)


train


train['year'] = pd.to_datetime(train['date']).dt.year
train['month'] = pd.to_datetime(train['date']).dt.month
train['day'] = pd.to_datetime(train['date']).dt.day
train.drop(columns=['date'], inplace=True)


train = pd.get_dummies(train, columns=['country', 'store', 'product'])


train


X = train.drop(['id', 'num_sold'], axis=1)
y = train['num_sold']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



X_train_scaled





from sklearn.linear_model import Ridge

model = Ridge(tol=1e-2, max_iter=1000000)
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_percentage_error

y_pred = model.predict(X_test_scaled)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'mean_absolute_percentage_error: {mape}')




