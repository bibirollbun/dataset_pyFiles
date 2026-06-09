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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_train.head(3)


df_train.info()


df_train.dropna(inplace=True)
df_train.head(3)


df_train['Episode_Title'].unique()


liste = df_train['Episode_Title'].str.split(' ')
df_train['Episode_Title'] = liste
df_train.head(3)


df_train['Episode_Title'] = df_train['Episode_Title'].str[1]


df_train.drop(columns = ['id','Podcast_Name'],inplace = True)
df_train.head(3)


print(df_train['Genre'].unique())
print(df_train['Publication_Day'].unique())
print(df_train['Publication_Time'].unique())
print(df_train['Episode_Sentiment'].unique())


df_train = pd.get_dummies(df_train, columns = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first = True)
df_train.head(3)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df_train = df_train.astype('float32')
df_train.info()


df_train['Listening_Time_minutes']=df_train['Listening_Time_minutes'].astype('int64')


from sklearn.linear_model import LinearRegression
y = df_train['Listening_Time_minutes']
x = df_train.drop(columns = ['Listening_Time_minutes'])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42 , train_size = 0.85) 

lr = LinearRegression()

model_lr_1 = lr.fit(x_train,y_train)

print(model_lr_1.score(x_test,y_test))

lr = LinearRegression()

main_model_lr_1 = lr.fit(x,y)


df_train.describe()


columns = ['Episode_Title','Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads','Listening_Time_minutes']
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(3,2,figsize=(16,16))
fig.suptitle('OUTLIERS')
plt.style.use('Solarize_Light2')
y = 0


for i in columns:
    a,b = divmod(y,3)
    ax[b][a].boxplot(df_train[i])
    ax[b][a].set_title(i)
    y+=1
        
    
plt.show()


deneme1_data = df_train.copy()

degisken = []

for i in columns:
    q1 = deneme1_data[i].quantile(.25)
    q3 = deneme1_data[i].quantile(.75)
    mask = deneme1_data[i].between(q1, q3)
    iqr = deneme1_data.loc[mask, i]
    list_iqr = list(iqr.index)
    degisken.extend(list_iqr)
degisken = list(set(degisken))
len(degisken)


deneme1_data = deneme1_data.loc[degisken]


from sklearn.linear_model import LinearRegression
y = deneme1_data['Listening_Time_minutes']
x = deneme1_data.drop(columns = ['Listening_Time_minutes'])

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42 , train_size = 0.85) 

lr = LinearRegression()

model_lr_2 = lr.fit(x_train,y_train)

print(model_lr_2.score(x_test,y_test))

lr = LinearRegression()

main_model_lr_2 = lr.fit(x,y)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_test.head(3)


df_test.info()


df_test['Episode_Length_minutes'] = df_test['Episode_Length_minutes'].fillna(df_test['Episode_Length_minutes'].mean())
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].fillna(df_test['Guest_Popularity_percentage'].mean())
df_test.info()


liste = df_test['Episode_Title'].str.split(' ')
df_test['Episode_Title'] = liste
df_test.head(3)


idler = df_test['id']


df_test['Episode_Title'] = df_test['Episode_Title'].str[1]


df_test.drop(columns = ['id','Podcast_Name'],inplace = True)
df_test.head(3)


df_test = pd.get_dummies(df_test, columns = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first = True)
df_test.head(3)


df_test = df_test.astype('float32')
df_test.info()


data_predict_1 = pd.DataFrame()
a = main_model_lr_1.predict(df_test)
data_predict_1['id'] = idler
data_predict_1['Listening_Time_minutes'] = a
data_predict_1


data_predict_1.to_csv('podcast_predict_1',index=False)


data_predict_2 = pd.DataFrame()
a = main_model_lr_2.predict(df_test)
data_predict_2['id'] = idler
data_predict_2['Listening_Time_minutes'] = a
data_predict_2


data_predict_2.to_csv('podcast_predict_2',index=False)




