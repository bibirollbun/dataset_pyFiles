# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv')
df_test = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv')
df_solutions = pd.read_csv('/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv')
df =df_train


df.head()


df.info()


df.describe()


df['airline'].value_counts()


df.price.hist()
plt.show()


label = LabelEncoder()
df_train['airline'] = label.fit_transform(df_train['airline'])
df_train['source_city'] = label.fit_transform(df_train['source_city'])
df_train['destination_city'] = label.fit_transform(df_train['destination_city'])
df_train['class'] = label.fit_transform(df_train['class'])
df_train['stops'] = label.fit_transform(df_train['stops'])
df_train['departure_time'] = label.fit_transform(df_train['departure_time'])
df_train['arrival_time'] = label.fit_transform(df_train['arrival_time'])


df_train.head()


df_test['airline'] = label.fit_transform(df_test['airline'])
df_test['source_city'] = label.fit_transform(df_test['source_city'])
df_test['destination_city'] = label.fit_transform(df_test['destination_city'])
df_test['class'] = label.fit_transform(df_test['class'])
df_test['stops'] = label.fit_transform(df_test['stops'])
df_test['departure_time'] = label.fit_transform(df_test['departure_time'])
df_test['arrival_time'] = label.fit_transform(df_test['arrival_time'])


df_test.head()


X_train = df_train.drop('price', axis=1)
y_train = df_train['price']


y_train.head()


train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)


num_attribs=['airline', 'stops', 'class', 'duration', 'days_left','source_city', 'departure_time', 'arrival_time','destination_city']
cat_attribs=['flight']

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])


X_prep = full_pipeline.fit_transform(X_train)
X_prep.toarray()[0:5,:]


X_test=test_set.drop('price', axis=1)
y_test=test_set['price'].copy()
X_test_prep = full_pipeline.transform(X_test)


RF_model = RandomForestRegressor()
RF_model.fit(X_prep, y_train)


y_predicted = RF_model.predict(X_test_prep)


test_prep = full_pipeline.transform(df_test)
predict_price = RF_model.predict(test_prep)


df_solutions['price'] = predict_price
df_solutions


df_solutions.to_csv('submission.csv', index=False)




