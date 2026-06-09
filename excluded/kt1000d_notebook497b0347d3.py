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


import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
#from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_log_error

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
MM=MinMaxScaler()
sc = StandardScaler()
def process_df(df):
    #df=pd.get_dummies(df, columns=['Embarked'])
    #df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex']=df['Sex'].map({'male':0, 'female':1})
    df['Age']=sc.fit_transform(df[['Age']])
    df['HW']=df['Height']*df['Weight']
    df['HW']=sc.fit_transform(df[['HW']])
    df['bm']=df['Weight'].div(df['Height'])
    df['Bmi']=df['bm'].div(df['Height'])
    df['Rohrer']=df['Bmi'].div(df['Height'])
    df['Rohrer']=sc.fit_transform(df[['Rohrer']])
    df['Bmi']=sc.fit_transform(df[['Bmi']])
    df['bm']=sc.fit_transform(df[['bm']])
    df['Weight']=sc.fit_transform(df[['Weight']])
    df['Height']=sc.fit_transform(df[['Height']])
    df['Duration']=sc.fit_transform(df[['Duration']])
    df['Heart_Rate']=sc.fit_transform(df[['Heart_Rate']])
    df['Body_Temp']=sc.fit_transform(df[['Body_Temp']])
    
    #df['Pclass']=MM.fit_transform(df[['Pclass']])
    #df["SiPa"]=df["SibSp"]+df["Parch"]
    df = df.drop(['id','bm','Height'], axis=1)
    return df

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
#print(train_df.describe())

#sns.scatterplot(x="Age", y="Calories",hue="Sex", data=train_df)
#sns.scatterplot(x="Weight", y="Height",hue="Sex", data=train_df)
#id Sex Age Height Weight Duration Heart_Rate Body_Temp Calories

#print(train_df.describe())
y=train_df["Calories"]
RS=46
X=process_df(train_df)
sns.scatterplot(x="Rohrer", y="Calories",hue="Sex", data=train_df)
df_corr = X.corr()
print(df_corr)
X = X.drop(['Calories'], axis=1)
X_test=process_df(test_data)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
#model1 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RS)
#model1.fit(train_X, train_y)
#y_test = model1.predict(val_X)
#print(mean_squared_log_error(val_y, y_test, squared=False),"RMSLE")
#0.2320174620497059 RMSLE

model2 = DecisionTreeRegressor(random_state=RS)
model2.fit(train_X, train_y)
y_test = model2.predict(val_X)
print(mean_squared_log_error(val_y, y_test, squared=False),"RMSLE")
#0.09177060801215493 RMSLE


#model3 = LogisticRegression(solver='liblinear', random_state=RS)
#model3.fit(train_X, train_y)
#y_test = model3.predict(val_X)
#print(mean_squared_log_error(val_y, y_test, squared=False),"RMSLE")
#交差検証
#folds = KFold(n_splits=3)

#for fold_, (trn_, val_) in enumerate(folds.split(train_X, train_y)):

    #trn_x, trn_y = train_X.iloc[trn_], train_y.iloc[trn_]
    #val_x, val_y = train_X.iloc[val_], train_y.iloc[val_]
    #model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RS)
    #model.fit(trn_x, trn_y)
    #y_test = model.predict(val_x)
    #print(mean_absolute_error(val_y, y_test),"RFC")
    #model = LogisticRegression(solver='liblinear', random_state=RS)
    #model.fit(trn_x, trn_y)
    #y_test = model.predict(val_x)
    #print(mean_absolute_error(val_y, y_test),"LR")
    #model = DecisionTreeRegressor(random_state=RS)
    #model.fit(trn_x, trn_y)
    #y_test = model.predict(val_x)
    #print(mean_absolute_error(val_y, y_test),"DTR")
    #pd.Series(y_test).value_counts()

model = DecisionTreeRegressor(random_state=RS)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Calories': predictions})
output.to_csv('submission.csv', index=False)


