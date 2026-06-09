# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as Sequential
from tensorflow.keras.layers import Dense,Dropout

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
X_test=pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")


df.shape


df.shape


df.isnull().sum()


# visualizing null values with heatmap
sns.heatmap(df.isnull(),cmap='summer')


df.duplicated().sum()


df.info()


catagorical=df.select_dtypes(include=['object']).columns
numerical=df.select_dtypes(exclude=['object']).columns
print(numerical)


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')


columns_to_impute = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress']
df[columns_to_impute]=imputer.fit_transform(df[columns_to_impute])


columns_cata=['Profession','Dietary Habits','Degree']
imputer_1=SimpleImputer(strategy='most_frequent')
df[columns_cata]=imputer_1.fit_transform(df[columns_cata])


df.isnull().sum()


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
for i in df.columns:
    if df[i].dtype=='object':
        df[i]=label.fit_transform(df[i])


df.drop(['id'],axis=1,inplace=True)


X=df.drop(columns=['Depression'])
y=df['Depression']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


rf=RandomForestClassifier(n_estimators=750,bootstrap=True,max_samples=0.7,max_features=0.6)
rf.fit(X_train,y_train)


y_pred=rf.predict(X_test)


accuracy_score(y_test,y_pred)


from mlxtend.plotting import plot_decision_regions


X_test.to_numpy().ravel()


ada= AdaBoostClassifier(n_estimators=750,learning_rate=0.1)
ada.fit(X_train,y_train)


y_pred=ada.predict(X_test)


accuracy_score(y_test,y_pred)


bagging=BaggingClassifier(n_estimators=500,bootstrap=False,bootstrap_features=True,max_samples=0.5)


bagging.fit(X_train,y_train)


y_pred=bagging.predict(X_test)
accuracy_score(y_test,y_pred)


import keras as Sequential
from tensorflow.keras.layers import Dense,Dropout


model=Sequential( 
    [
        Dense(64,activation='sigmoid'),
        Dense(32,activation='relu'),
        Dense(16,activation='relu'),
        Dense(1,activation='relu')
    ]
)




