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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col = 'id')
print(df_train.shape)
df_train.sample(5)


df_train.isnull().sum()


X_train = df_train.iloc[:, 0:10]
y_train = df_train.iloc[:, -1]
print(X_train.shape, y_train.shape)
#print(type(y_train))


X_train['Episode_Length_minutes'].fillna(X_train['Episode_Length_minutes'].mean(), inplace = True)
X_train['Guest_Popularity_percentage'].fillna(X_train['Guest_Popularity_percentage'].mean(), inplace = True)
X_train['Number_of_Ads'].fillna(X_train['Number_of_Ads'].mode()[0], inplace = True)

print(X_train.isnull().sum())



duplicate_columns = X_train.apply(lambda x: x.duplicated().any())

print(duplicate_columns)


X_train.nunique()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


transformer = ColumnTransformer(transformers = [
    ('tnf1', OrdinalEncoder(categories = [['Negative', 'Neutral', 'Positive']]), ['Episode_Sentiment']),
    ('tnf2', OneHotEncoder(sparse = False, drop = 'first'), 
     ['Podcast_Name','Genre','Publication_Day','Publication_Time'])
],
remainder = 'passthrough')


X_train.drop(['Episode_Title'], inplace = True, axis = 1)
X_train.sample()


X_train_transformed = transformer.fit_transform(X_train)
print(type(X_train_transformed))
print(X_train_transformed[0])

y_train = y_train.values.ravel()


X_train_transformed.shape


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train_transformed, y_train)


X_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
id = X_test['id']

X_test.drop(['Episode_Title','id'], inplace = True, axis = 1)



print(X_test.isnull().sum())


X_test['Episode_Length_minutes'].fillna(X_test['Episode_Length_minutes'].mean(), inplace = True)
X_test['Guest_Popularity_percentage'].fillna(X_test['Guest_Popularity_percentage'].mean(), inplace = True)


X_test_tranformed = transformer.transform(X_test)


y_pred = model.predict(X_test_tranformed)


Listening_Time_minutes = pd.Series(y_pred, name = 'Listening_Time_minutes')


final_df = pd.concat([id, Listening_Time_minutes], axis = 1)
final_df.head()


final_df.to_csv('submission.csv', index = False)


X_test_tranformed.shape




