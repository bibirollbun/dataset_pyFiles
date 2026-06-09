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


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.fillna(test.mean(),inplace=True)


train


import seaborn as sns 
sns.scatterplot(data=train,x='temparature',y='cloud',hue='rainfall')



x=train.drop(columns=['id','day','rainfall'])
y=train['rainfall']
z=test.drop(columns=['id','day'])



#decision tree classifer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,mean_squared_error
d=DecisionTreeClassifier(max_depth=10)
d.fit(x,y)
pred=d.predict(z)
trainpred=d.predict(x)
print('accuracy score on train data by trees:',accuracy_score(y,trainpred))



#Logistic regresssion 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
lr.fit(x,y)
pred2=lr.predict(z)
trainpred2=lr.predict(x)
print('accurac score on train data by logistic regression:',accuracy_score(y,trainpred2))





from sklearn.ensemble import BaggingClassifier
a=BaggingClassifier(
    estimator=d,
    n_estimators=21,
    max_samples=0.7,
    bootstrap=True,
    random_state=42
)
a.fit(x,y)
pred3=a.predict(z)
trainpred3=a.predict(x)
print('accuracy score on train data by decision tree bagging classifier:',accuracy_score(y,trainpred3))


#bagging classifier for logistic regression
from sklearn.ensemble import BaggingClassifier
a=BaggingClassifier(
    estimator=lr,
    max_samples=0.7,
    bootstrap=True,
    random_state=42
)
a.fit(x,y)
pred4=a.predict(z)
trainpred4=a.predict(x)
print('accuracy score on train data by decision tree bagging classifier:',accuracy_score(y,trainpred4))





from sklearn.ensemble import BaggingClassifier
a=BaggingClassifier(
    estimator=d,
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)
a.fit(x,y)
pred5=a.predict(z)
trainpred5=a.predict(x)
print('accuracy score on train data by decision tree bagging classifier:',accuracy_score(y,trainpred5))


a=pd.DataFrame({
    'id':test['id'],
    'rainfall':pred5
})
a



a.to_csv('submission.csv',index=False)

