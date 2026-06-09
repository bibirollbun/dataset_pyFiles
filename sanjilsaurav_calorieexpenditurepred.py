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


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


df.head()


df.describe()


df.info()


import matplotlib.pyplot as plt
import seaborn as sns


df['Sex'].unique()


sex_map = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_map)


df.head()


df.info()


df.drop('id', axis=1, inplace=True)


df.head()


sns.histplot(df['Height'], bins=50, kde=True)


sns.histplot(df['Age'], bins=50, kde=True)


sns.histplot(df['Weight'], bins=50, kde=True)


sns.histplot(df['Duration'], bins=50, kde=True)


sns.histplot(df['Heart_Rate'], bins=50, kde=True)


sns.histplot(df['Body_Temp'], bins=50, kde=True)


sns.histplot(df['Calories'], bins=50, kde=True)


from sklearn.model_selection import train_test_split


X = df.drop('Calories', axis=1)
y = df['Calories']
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=105, shuffle=True)


X_train.head()


y_train.head()


from sklearn.linear_model import LinearRegression


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
print(lin_model.score(X_test, y_test))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test['Sex'] = df_test['Sex'].map(sex_map)


df_test.head()


Inp_Feat = df_test.drop('id', axis=1)


Inp_Feat.head()


y_pred_lin = lin_model.predict(Inp_Feat)
y_pred_lin


for i in range(len(y_pred_lin)):
    if y_pred_lin[i] < 0:
        y_pred_lin[i] = abs(y_pred_lin[i])


res = {'id':df_test['id'], 'Calories': y_pred_lin}
res_df = pd.DataFrame(res)


res_df.head()


res_df.to_csv('CalPred.csv', index=False)


df_test.loc[23]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
scaled_df.head()


X_sc = scaled_df.drop('Calories', axis=1)
y_sc = scaled_df['Calories']
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_sc, y_sc, test_size=0.2, random_state=105, shuffle=True)


lin_model.fit(X_train_sc, y_train_sc)
print(lin_model.score(X_test_sc, y_test_sc))


scaled_test = scaler.fit_transform(Inp_Feat)
scaled_test_df = pd.DataFrame(scaled_test, columns = Inp_Feat.columns)
scaled_test_df.head()


y_pred_lin_sc = lin_model.predict(scaled_test_df)
y_pred_lin_sc


X_train_sc.head()


cnt = 0
for i in y_pred_lin_sc:
    if i<=0:
        cnt += 1
print(cnt)


from sklearn.ensemble import RandomForestRegressor


lst = [0.9962913751403689]


rf_model = RandomForestRegressor(n_estimators=70, n_jobs=5)
rf_model.fit(X_train, y_train)
scr = rf_model.score(X_test, y_test)
lst.append(scr)
print(lst)


y_pred_rf = rf_model.predict(Inp_Feat)
y_pred_rf


cnt = 0
for i in y_pred_rf:
    if i<=0:
        cnt += 1
print(cnt)


rf_res = {'id': df_test['id'], 'Calories': y_pred_rf}
rf_res_df = pd.DataFrame(rf_res)
rf_res_df.head()


rf_res_df.to_csv('CalPredRF2.csv', index=False)




