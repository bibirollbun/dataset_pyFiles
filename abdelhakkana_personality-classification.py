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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


df1=pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
df2=pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')
df3=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


#df4=pd.concat([df1,df2],ignore_index=True)


#df4


#df4.duplicated().sum()


#df4.drop_duplicates(inplace=True)


#df4.duplicated().sum()


#df4


df3


df3.drop('id',inplace=True,axis=1)


df3


#df5=pd.concat([df3,df4],ignore_index=True)


#df5


#df5.duplicated().sum()


#df5.drop_duplicates(inplace=True)


#df5


X=df3.drop('Personality',axis=1)
y=df3['Personality']
train=X


train.info()


train.isnull().sum()


#df5.groupby('Personality').describe()


categorical=[i for i in X if X[i].dtype=='object']
numerical=[j for j in X if X[j].dtype in ['float64']]


categorical_transformer=Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='not_recorded')),('ord',OrdinalEncoder())])
numerical_transformer=SimpleImputer(strategy='median')
preprocessor=ColumnTransformer(transformers=[('cat',categorical_transformer,categorical),('num',numerical_transformer,numerical)])


X=preprocessor.fit_transform(X)


st=StandardScaler()
sc=MinMaxScaler()


X1=st.fit_transform(X)
X2=sc.fit_transform(X)


lb=LabelEncoder()
y1=lb.fit_transform(y)


X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.12,stratify=y1,random_state=11)
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.12,stratify=y1,random_state=11)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y1,test_size=0.12,stratify=y1,random_state=11)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


clf1=LogisticRegression(random_state=33)
clf1.fit(X_train,y_train)
y1_pred=clf1.predict(X_test)
accuracy_score(y1_pred,y_test)


clf2=SVC(random_state=33)
clf2.fit(X_train,y_train)
y2_pred=clf2.predict(X_test)
accuracy_score(y2_pred,y_test)


clf3=RandomForestClassifier(random_state=33)
clf3.fit(X_train,y_train)
y3_pred=clf3.predict(X_test)
accuracy_score(y3_pred,y_test)


clf4=GradientBoostingClassifier(random_state=33)
clf4.fit(X_train,y_train)
y4_pred=clf4.predict(X_test)
accuracy_score(y4_pred,y_test)


clf5=DecisionTreeClassifier(random_state=33)
clf5.fit(X_train,y_train)
y5_pred=clf5.predict(X_test)
accuracy_score(y5_pred,y_test)


clf6=KNeighborsClassifier()
clf6.fit(X_train,y_train)
y6_pred=clf6.predict(X_test)
accuracy_score(y6_pred,y_test)


clf7=XGBClassifier(random_state=33)
clf7.fit(X_train,y_train)
y7_pred=clf7.predict(X_test)
accuracy_score(y7_pred,y_test)


clf8=CatBoostClassifier(random_state=3,verbose=0)
clf8.fit(X_train,y_train)
y8_pred=clf8.predict(X_test)
accuracy_score(y8_pred,y_test)


clf9=LGBMClassifier(random_state=30)
clf9.fit(X_train,y_train)
y9_pred=clf9.predict(X_test)
accuracy_score(y9_pred,y_test)


test


test.isnull().sum()


testt=test[train.columns]


testty=preprocessor.transform(testt)


predictions=clf8.predict(testty)


predictions


predictions=lb.inverse_transform(predictions)


predictions


output=pd.DataFrame({'id':test.id,'Personality':predictions})
output


submission=output.to_csv('submission.csv',index=False)




