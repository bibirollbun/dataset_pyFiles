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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error




train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')



print(train_data.head())

print(train_data.describe())

print(train_data.isnull().sum())




train_data['num_sold'].fillna(train_data['num_sold'].median(), inplace=True)


combined_data = pd.concat([train_data, test_data], keys=['train', 'test'])


combined_data['date'] = pd.to_datetime(combined_data['date'])
combined_data['year'] = combined_data['date'].dt.year
combined_data['month'] = combined_data['date'].dt.month
combined_data['day'] = combined_data['date'].dt.day
combined_data['day_of_week'] = combined_data['date'].dt.dayofweek


categorical_features = ['country', 'store', 'product']
combined_data = pd.get_dummies(combined_data, columns=categorical_features)


train_data = combined_data.loc['train']
test_data = combined_data.loc['test']




X = train_data.drop(columns=['id', 'date', 'num_sold'])
y = train_data['num_sold']


X_test = test_data.drop(columns=['id', 'date', 'num_sold'])




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)




y_val_pred = model.predict(X_val)


mape = mean_absolute_percentage_error(y_val, y_val_pred)
print(f'Validation MAPE: {mape:.4f}')




test_predictions = model.predict(X_test)




submission = pd.DataFrame({
    'id': test_data['id'],
    'num_sold': test_predictions
})


submission.to_csv('submission.csv', index=False)





