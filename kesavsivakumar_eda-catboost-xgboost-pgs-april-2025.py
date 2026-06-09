# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train= pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")


df_train.head()


df_train['Podcast_Name'].value_counts()


df_train['Episode_Title'].unique()


df_train['Genre'].value_counts()


df_train.groupby(['Genre']).agg(max)


df_train.groupby(['Genre']).agg(min)


df_train.groupby(['Genre'])['Listening_Time_minutes'].agg("mean")


df_train.groupby(['Genre'])['Listening_Time_minutes'].agg("median")


df_train.info()


import seaborn as sns
%matplotlib inline


plt.xticks(rotation=75)

sns.set_theme(rc={'figure.figsize':(60,30)})
sns.boxplot(x="Podcast_Name", y="Listening_Time_minutes", data=df_train)


sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Genre", y="Listening_Time_minutes", data=df_train)


sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Publication_Day", y="Listening_Time_minutes", data=df_train)


sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Publication_Time", y="Listening_Time_minutes", data=df_train)



sns.set_theme(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Episode_Sentiment", y="Listening_Time_minutes", data=df_train)


## label encode columns







#calculate the correlation matrix on the numeric columns
corr = df_train.select_dtypes('number').corr()

# plot the heatmap
sns.heatmap(corr)


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split 


y = df_train['Listening_Time_minutes'].values
fin_columns = [ ]
for col in df_train.columns:
    if col not in ['Listening_Time_minutes','id']:
        fin_columns.append(col)
X = df_train.loc[:, fin_columns].values
X_train,X_val,y_train,y_val =  train_test_split(X,y,test_size= 0.25)


cat_feature_idxs = [] 
for i in range(X.shape[1]):
    if 'str' in str(type(X_train[0,i])):
        print(type(X_train[0,i]))
        cat_feature_idxs.append(i)


model=CatBoostRegressor(iterations=4000, depth=10, learning_rate=0.01, loss_function='RMSE')


model.fit(X_train,y_train ,plot=True,use_best_model=True,eval_set = (X_val,y_val),cat_features =cat_feature_idxs)


df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")


df_test


ids = df_test['id'].values


X_test = df_test.loc[:, df_test.columns != 'id'].values


pred=model.predict(X_test)


df_dict = {'id':ids,'Listening_Time_minutes':pred}
df_subm = pd.DataFrame(df_dict)


df_subm


df_subm.to_csv('submission.csv',index=False)




