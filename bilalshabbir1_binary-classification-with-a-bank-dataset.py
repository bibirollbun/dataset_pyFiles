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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline


df_train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


df_train.head(5)



df_test.head(5)


df_train.isnull().sum()


df_test.isnull().sum()


print(df_train.duplicated().sum())
print(df_test.duplicated().sum())


num_col=df_train[['id','age','balance','day','duration','campaign','pdays','previous','y']]


num_col.corr()


sns.heatmap(num_col.corr(),cmap="Blues")


num_col2=df_test[['id','age','balance','day','duration','campaign','pdays','previous']]
sns.heatmap(num_col2.corr(),cmap="Blues")


df_train.shape


df_train.head()


X=df_train.iloc[:,0:17]
y=df_train.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
X.columns


trf1=ColumnTransformer([
    ('OneHotEncoder',OneHotEncoder(sparse_output=False,drop='first'),[2,5,7,8,9,16]),
    ('OrdinalEncoder',OrdinalEncoder(),[3,4,11]),# marital,education & month
    ("StandardScaler",StandardScaler(),[0,1,6,10,12,13,14,15])
],remainder='passthrough')


lr=LogisticRegression(l1_ratio= 0.25,n_jobs=-1,penalty= 'l2')
dt=DecisionTreeClassifier(splitter='best',max_depth=15)
rf=RandomForestClassifier(max_depth=25,bootstrap=False,n_estimators=130,max_features=0.5,n_jobs=-1,)


trf1.fit(X_train,y_train)


pipe=Pipeline([
    ('trf1',trf1),
    ('rf',rf)
])


pipe.fit(X_train,y_train)


accuracy_score(y_test,pipe.predict(X_test))


roc_auc_score(y_test,pipe.predict(X_test))


from sklearn.model_selection import GridSearchCV


param_grid={
    'rf__n_estimators':[100,50,200], 
    'rf__max_depth':[5,7,10,15], 
    'rf__min_samples_split':[10000,20000],
    'rf__min_samples_leaf':[1000,5000]
}
param_grid1={
    'lr__penalty':['l2','l1''elasticnet'],
    'lr__n_jobs':[-1],
    'lr__l1_ratio':[0.25,0.5,0.75,1.0]
}
param_grid2={
    'dt__splitter':['random','best'],'dt__max_depth':[10,15,20,25]
}


grid=GridSearchCV(pipe,param_grid2,scoring='accuracy')


#grid.fit(X_train,y_train)


#grid.best_params_


#grid.best_score_


X=df_test.iloc[:,0:17]
X



y_pred=(pipe.predict(X))
y_pred


submission=pd.DataFrame({
    'id':df_test['id'],
    'y':y_pred
})


submission


submission.to_csv('submission.csv',index=False)


df=pd.read_csv('submission.csv')

