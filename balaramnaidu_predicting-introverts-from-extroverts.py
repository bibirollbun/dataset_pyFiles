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


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.isnull().sum()


df.info()


df


df = df.dropna()
df


df.describe()


numerical_cols = df.describe().columns
numerical_cols = numerical_cols[1:]
numerical_cols


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
scaler = StandardScaler()
df1 = df.copy()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df


le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])
df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
df


cols = df.columns
cols = cols[1:-1]
cols


x = df[cols]
x


y = df['Personality']
y


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train


y_train


model.fit(x_train, y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_train, model.predict(x_train))


accuracy_score(y_test, y_pred)


confusion_matrix(y_test, y_pred)


y_test


df2 = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'Mismatch': y_test != y_pred}, index=y_test.index)
matched_df = df1.loc[df1.index.isin(df2.index)]
print(matched_df.columns)
df2 = df2[df2['Mismatch'] == True]
df2


matched_df = df1.loc[df1.index.isin(df2.index)]
matched_df


df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df


df.isnull().sum()


df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
df


df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df.fillna(0, inplace=True)
df


x = df[cols]
x


model.predict(x)


submission_df = pd.DataFrame()
submission_df['id'] = df['id']  # Use Series directly
submission_df['Personality'] = model.predict(x)
le = LabelEncoder().fit(['Extrovert', 'Introvert'])
submission_df['Personality'] = le.inverse_transform(submission_df['Personality'])
submission_df


submission_df.to_csv('submission.csv', index=False)
print('done')
submission_df

