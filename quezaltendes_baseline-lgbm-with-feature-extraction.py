# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
ss = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train


train.describe()


test


y_train = train['Listening_Time_minutes']
X_train = train.drop(columns=['id', 'Listening_Time_minutes'])


X_train_num_visual = X_train.select_dtypes(include=['int64', 'float64'])


import matplotlib.pyplot as plt
for column in X_train_num_visual.columns:
    plt.hist(X_train_num_visual[column], bins=100)
    plt.title(column)
    plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(
    pd.concat([X_train_num_visual, y_train], axis=1).corr(),
    annot=True,  
    fmt=".4f",      
    cmap="coolwarm", 
    vmin=-1, vmax=1, 
    linewidths=0.5
)
plt.title("Corr Matrix")
plt.show()


X_train_cb = pd.concat([train, pd.Series(np.random.rand(750000))], axis=1).dropna()
X_train_cb


train_data = Pool(data=(pd.concat([X_train_cb.drop(columns='Listening_Time_minutes')], axis=1)), label=X_train_cb['Listening_Time_minutes'], cat_features=['Podcast_Name',  'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


model = CatBoostRegressor(iterations=100, verbose=False)
model.fit(train_data)


importance = model.get_feature_importance()


lsit = {}
for i in range(len(importance)):
    lsit[importance[i]] = ((X_train_cb.drop(columns='Listening_Time_minutes')).columns)[i]


lsit


df = pd.DataFrame({
    'Feature': lsit.values(),
    'Importance': lsit.keys()
}).sort_values('Importance', ascending=False)


sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

colors = sns.color_palette("viridis", len(df))

barplot = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=df, 
    palette=colors,
)

plt.title('Feature importances', fontsize=16, pad=20)
plt.xlabel('Importances', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for p in barplot.patches:
    width = p.get_width()
    plt.text(
        width * 1.02, 
        p.get_y() + p.get_height()/2, 
        f'{width:.2f}', 
        va='center', 
        fontsize=9,
        color='black'
    )

plt.subplots_adjust(left=0.3)

plt.show()


X_train


X_train_ac = X_train.drop(columns=['Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


X_train_ac.isna().sum()


X_train_ac =  X_train_ac.fillna(X_train_ac.median())


test


ids = test['id']
X_test = test.drop(columns=['id', 'Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


X_test = X_test.fillna(X_test.median())


model = LGBMRegressor()


model.fit(X_train_ac, y_train)


y_test = model.predict(X_test)


ss


pd.DataFrame({'id': ids, 'Listening_Time_minutes': y_test}).to_csv('submission.csv', index=False)

