import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import warnings 
warnings.filterwarnings('ignore')


%%capture
!pip install morethemes


import morethemes as mt
mt.set_theme("minimal")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Посмотрим на train.csv
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()


# Посмотрим распределение данных и опишем их
df.describe()


columns = list(df.columns)
columns.remove('id')
columns.remove('rainfall')

fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(4*5, 5*5))

for id, col in enumerate(columns):
    axs[id // 5][id % 5].set_title(col)
    axs[id // 5][id % 5].violinplot(df[col])

plt.show()   


def scatter_line(col, data=df):
    fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(25, 3))

    columns_without = columns.copy()
    columns_without.remove(col)

    for id, c in enumerate(columns_without):
        axs[id].set_title(f"{c} {round(df[col].corr(df[c]), 2)}")
        axs[id].scatter(df[c], df[col])  

    plt.show()


def rain_notrain_hist(col, data=df):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    axs[0].hist(df[col], bins=16)

    axs[1].hist(df.loc[df['rainfall'] == 0, col], bins=16, color='g', alpha=0.5, label='not rain')
    axs[1].hist(df.loc[df['rainfall'] == 1, col], bins=16, alpha=0.5, label='rain')
    axs[1].legend()

    plt.show()


rain_notrain_hist('pressure')


df['pressure'].corr(df['rainfall'])


def scatter_line(col, data=df):
    fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(25, 3))

    columns_without = columns.copy()
    columns_without.remove(col)

    for id, c in enumerate(columns_without):
        axs[id].set_title(f"{c} {round(df[col].corr(df[c]), 2)}")
        axs[id].scatter(df[c], df[col])  

    plt.show()


scatter_line('pressure')


rain_notrain_hist('dewpoint')


df['dewpoint'].corr(df['rainfall'])


scatter_line('dewpoint')


# Сезон года из дня
print(df['day'].corr(df['rainfall']))
df['season'] = df['day'] // 92
print(df['season'].corr(df['rainfall']))


# Разница температуры
df['max-min'] = df['maxtemp'] - df['mintemp']
print(df['max-min'].corr(df['rainfall']))


# Значение погоды перемножим между собой и между давлением
df['pressure_humidity'] = df['humidity'] * df['pressure']
df['pressure_cloud'] = df['cloud'] * df['pressure']
df['pressure_sunshine'] = df['sunshine'] * df['pressure']
df['humidity_cloud'] = df['humidity'] * df['cloud']
df['humidity_sunshine'] = df['humidity'] * df['sunshine']
df['cloud_sunshine'] = df['cloud'] * df['sunshine']

df[['humidity', 'cloud', 'sunshine', 'pressure',  'rainfall']].corr()


df[['pressure_humidity', 'pressure_cloud', 'pressure_sunshine', 'humidity_cloud', 'humidity_sunshine', 'cloud_sunshine', 'rainfall']].corr()


# Направление ветра разобъем на 4 части
print(df['winddirection'].corr(df['rainfall']))
df['winddirection_label'] = df['winddirection'] // 90
print(df['winddirection_label'].corr(df['rainfall']))


from sklearn.preprocessing import StandardScaler, MinMaxScaler


X_train, X_test, y_train, y_test = train_test_split(df.drop(['id', 'rainfall'], axis=1), df['rainfall'], test_size=0.2, shuffle=True, random_state=42)


cat_columns = ['day', 'season', 'winddirection_label']
num_cols = X_train.drop(columns=cat_columns).columns.tolist()

scaler_dict = {}

for col in num_cols:
    scaler_dict[col] = StandardScaler()
    X_train[col] = scaler_dict[col].fit_transform(X_train[[col]])
    X_test[col] = scaler_dict[col].transform(X_test[[col]])


from statsmodels.stats.outliers_influence import variance_inflation_factor

df_for_vif = df.drop(['id', 'rainfall'], axis=1)

vif = pd.DataFrame()
vif["features"] = df_for_vif.columns
vif["VIF"] = [variance_inflation_factor(df_for_vif.values, i) for i in range(df_for_vif.shape[1])]

vif.sort_values(by='VIF', ascending=False)


drop_cols = [
    'id', 'rainfall',
    'maxtemp', 'mintemp', 'pressure', 'humidity_cloud', 'cloud_sunshine', 'humidity_sunshine', 'cloud_sunshine', 'day', 'winddirection',
    'pressure_humidity', 'pressure_cloud', 'pressure_sunshine'
]

df_for_vif = df.drop(columns=drop_cols, axis=1)

vif = pd.DataFrame()
vif["features"] = df_for_vif.columns
vif["VIF"] = [variance_inflation_factor(df_for_vif.values, i) for i in range(df_for_vif.shape[1])]

vif.sort_values(by='VIF', ascending=False)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    penalty='l1',
    solver='saga',
    random_state=42,
)

lr.fit(X_train[df_for_vif.columns], y_train)
print(roc_auc_score(lr.predict(X_test[df_for_vif.columns]), y_test))


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

test['winddirection'].fillna(test['winddirection'].mean(), inplace=True)

test['max-min'] = test['maxtemp'] - test['mintemp']

test_data = test[['temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed', 'max-min']]

for col in test_data.columns.tolist():
    test_data[col] = scaler_dict[col].transform(test_data[[col]])

test_data['season'] = test['day'] // 92
test_data['winddirection_label'] = test['winddirection'] // 90

predict = lr.predict(test_data[df_for_vif.columns])

pd.Series(data=predict, index=test['id']).to_csv('submission.csv')

