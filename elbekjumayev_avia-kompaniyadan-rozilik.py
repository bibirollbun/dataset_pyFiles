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


df=pd.read_csv('/kaggle/input/aviakompaniya/train_dataset.csv', index_col=0)


df.head(5)


from sklearn.preprocessing import LabelEncoder
l_e=LabelEncoder()
df['Gender']=l_e.fit_transform(df['Gender'].values)
df['Customer Type']=l_e.fit_transform(df['Customer Type'].values)
df['Type of Travel']=l_e.fit_transform(df['Type of Travel'].values)
df['Class']=l_e.fit_transform(df['Class'].values)



df.head(5)


df.info()


df['Arrival Delay in Minutes'].isnull().sum()


df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(0)


df['Arrival Delay in Minutes'].isnull().sum()


df.corrwith(df['satisfaction']).abs().sort_values(ascending=False)


df['satisfaction'].value_counts() # natijalar tengligini tekshirib ko'ramiz


x=df.drop('satisfaction',axis=1).values
y=df['satisfaction']


from sklearn.preprocessing import StandardScaler   # sonlarni normalizatsiya qilamiz
scaler=StandardScaler()
x=scaler.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)


y_predict=knn.predict(x_test)


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_predict)
print('Aniqlik darjasi: ',ac,'%')


from sklearn.model_selection import GridSearchCV
pg={'n_neighbors':np.arange(1,20)}
knn_gscv=GridSearchCV(knn,pg,cv=5)
knn_gscv.fit(x,y)


knn_gscv.best_params_




