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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_train.sample(5)


print(df_train.shape)
print(df_train.info())
print(df_train.isnull().sum())
print(df_train.describe())
print(df_train.duplicated().sum())


df_train.corr()['rainfall']



#import seaborn as sns
#sns.countplot(df_train['rainfall'])
df_train['rainfall'].value_counts().plot(kind='bar')



import matplotlib.pyplot as plt

plt.hist(df_train['temparature'])


import seaborn as sns

sns.distplot(df_train['temparature'])


sns.boxplot(df_train['humidity'])


#sns.pairplot(df_train)



# from ydata_profiling import ProfileReport
# prof = ProfileReport(df_train)
# prof.to_file('Report.html')


df_train.sample()


df_train.drop(['id', 'day', 'winddirection'], axis = 1, inplace = True)



X_train = df_train.drop(['rainfall'], axis = 1)
print(X_train)


y_train = df_train['rainfall']
y_train.head()


X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
print(X_train.head())
print(y_train.head())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
#y_train_scaled = scaler.transform(y_train)


from sklearn.linear_model import LogisticRegression


lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)


x_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(x_test.head())


id = x_test['id']
print(id.head())


x_test.drop(['id', 'day','winddirection'], axis = 1, inplace = True)
x_test.head()


x_test_scaled = scaler.transform(x_test)
print(x_test_scaled)


y_pred = lr.predict(x_test_scaled)
print(y_pred)


rainfall = pd.Series(y_pred, name = 'rainfall')
print(rainfall)


final_df = pd.concat([id, rainfall], axis = 1)
print(final_df.head())


##final_df.to_csv('submission1.csv', index = False)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

y_train = df_train['rainfall']

X_train = df_train.drop(['id', 'day', 'winddirection', 'rainfall'], axis = 1)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(df_test.head())


id = df_test['id']
X_test = df_test.drop(['id','day', 'winddirection'], axis = 1)

X_test.head()


y_pred = model.predict(X_test)
print(y_pred)


rainfall = pd.Series(y_pred, name = 'rainfall')
final_df = pd.concat([id, rainfall], axis = 1)

final_df.to_csv('submission.csv', index = False)




