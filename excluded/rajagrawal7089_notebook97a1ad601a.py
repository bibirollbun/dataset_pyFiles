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

# Load the training data
df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Load the test data
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


df.head()


df.info()


df.isnull().sum()


median = df['num_sold'].median()
median


df['num_sold'] = df['num_sold'].fillna(median)


df.isnull().sum()


df.duplicated()


df = df.drop_duplicates()


df.info(20)


df['date'] = pd.to_datetime(df['date'])


df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


df['country'] = df['country'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


df['store'] = df['store'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


df['product'] = df['product'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek


df


df = pd.get_dummies(df, columns=['country', 'store', 'product'], drop_first=True)


X = df.drop(columns=['date', 'num_sold'])  # Replace 'num_sold' with your target column name
y = df['num_sold']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)


from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_val, y_pred)
print(f'MAPE: {mape}')


test


test = test.drop_duplicates()


test['date'] = pd.to_datetime(test['date'])


test = test.apply(lambda y: y.str.strip() if y.dtype == "object" else y)


test['country'] = test['country'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


test['store'] = test['store'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


test['product'] = test['product'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)


test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['day_of_week'] = test['date'].dt.dayofweek


test


test = pd.get_dummies(test, columns=['country', 'store', 'product'], drop_first=True)


test.drop(columns=['date'], inplace=True)


test


test_pred = model.predict(test)


# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test['id'],  # Assuming you have an 'id' column in your test data
    'num_sold': test_pred  # Replace 'num_sold' with your target variable name
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)




