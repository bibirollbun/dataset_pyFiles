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


df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
# df.set_index("id",inplace=True)
df


df.info()
# df["windspeed"].unique()


df.corr()


import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df.corr())


# maximum and minimum temperatures, atmospheric pressure, relative humidity, and wind speed
plt.hist(df["windspeed"], bins=30, color='skyblue', edgecolor='black')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



df["temparature"]=(df["mintemp"]+df["maxtemp"])/2
df["pressure"]=df["pressure"]%1000
X=df.drop(columns=["rainfall","mintemp","maxtemp","id","day"])
y=df["rainfall"].astype(float)
X_train, X_test,y_train, y_test = train_test_split(X,y , 
                                   random_state=84,  
                                   test_size=0.25,  
                                   shuffle=True) 


X_train


logis=LogisticRegression()
knn=KNeighborsClassifier()


logis.fit(X_train,y_train)


X_test


result=logis.predict(X_test)


accuracy_score(y_test,result)*100


f1_score(y_test, result)*100


