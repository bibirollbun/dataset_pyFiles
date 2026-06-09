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


df= pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


# remove ID col
df= df.iloc[:,1:]


df.head()


df.info()


df.describe()


# label encode target 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Target']=encoder.fit_transform(df['Target'])


df.head()


X = df.drop(columns=['Target'])
y = df['Target']


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


clf1= LogisticRegression()
clf2= RandomForestClassifier()
clf3= KNeighborsClassifier()


estimators= [('lr',clf1),('rf',clf2),('svc',clf3)]


for estimator in estimators:
    x= cross_val_score(estimator[1],X,y,cv=5,scoring = 'accuracy')
    print(estimator[0],np.round(np.mean(x),2))


from sklearn.ensemble import VotingClassifier


# Hard voting
vc= VotingClassifier(estimators=estimators)
x= cross_val_score(estimator[1],X,y,cv=5,scoring = 'accuracy')
print(np.round(np.mean(x),2))


# soft voting
vc1= VotingClassifier(estimators=estimators,voting='soft')
x= cross_val_score(estimator[1],X,y,cv=5,scoring = 'accuracy')
print(np.round(np.mean(x),2))


# # adding weights
# for i in range(1,4):
#     for j in range(1,4):
#         for k in range(1,4):
#             vc= VotingClassifier(estimators=estimators,voting='soft',weights=[i,j,k])
#             x= cross_val_score(estimator[1],X,y,cv=10,scoring = 'accuracy')
#             print('for i={},j={},k={}'.format(i,j,k),np.round(np.mean(x),2))


from sklearn.datasets import make_classification
from sklearn.svm import SVC
X,y = make_classification(n_samples=1000,n_features=20,n_informative=15,n_redundant=5,random_state=2)

svm1 = SVC(probability=True, kernel='poly', degree=1)
svm2 = SVC(probability=True, kernel='poly', degree=2)
svm3 = SVC(probability=True, kernel='poly', degree=3)
svm4 = SVC(probability=True, kernel='poly', degree=4)
svm5 = SVC(probability=True, kernel='poly', degree=5)

estimators = [('svm1',svm1),('svm2',svm2),('svm3',svm3),('svm4',svm4),('svm5',svm5)]

for estimator in estimators:
    x = cross_val_score(estimator[1],X,y,cv=10,scoring='accuracy')
    print(estimator[0],np.round(np.mean(x),2))


vc1 = VotingClassifier(estimators=estimators,voting='soft')
x = cross_val_score(vc1,X,y,cv=10,scoring='accuracy')
print(np.round(np.mean(x),2))


train= pd.read_csv('/kaggle/input/openintro-possum/possum.csv')


train.shape



train.info()


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
# train['Pop_encode']=encoder.fit_transform(train['Pop'])
train['sex'] = train['sex'].fillna('Unknown')
train['sex_encode']=encoder.fit_transform(train['sex'])


train = train.dropna()


train=train.drop(['Pop','sex','case','site','age'],axis=1)


X=train.drop('sex_encode',axis=1)
y=train['sex_encode']


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


lr = LinearRegression()
dt = DecisionTreeRegressor()
svr = SVR()


estimators = [('lr',lr),('dt',dt),('svr',svr)]
     

for estimator in estimators:
  scores = cross_val_score(estimator[1],X,y,scoring='r2',cv=10)
  print(estimator[0],np.round(np.mean(scores),2))



from sklearn.ensemble import VotingRegressor
     

vr = VotingRegressor(estimators)
scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
print("Voting Regressor",np.round(np.mean(scores),2))


for i in range(1,4):
  for j in range(1,4):
    for k in range(1,4):
      vr = VotingRegressor(estimators,weights=[i,j,k])
      scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
      print("For i={},j={},k={}".format(i,j,k),np.round(np.mean(scores),2))


# using the same algorithm

dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=3)
dt3 = DecisionTreeRegressor(max_depth=5)
dt4 = DecisionTreeRegressor(max_depth=7)
dt5 = DecisionTreeRegressor(max_depth=None)
     

estimators = [('dt1',dt1),('dt2',dt2),('dt3',dt3),('dt4',dt4),('dt5',dt5)]
     

for estimator in estimators:
  scores = cross_val_score(estimator[1],X,y,scoring='r2',cv=10)
  print(estimator[0],np.round(np.mean(scores),2))



vr = VotingRegressor(estimators)
scores = cross_val_score(vr,X,y,scoring='r2',cv=10)
print("Voting Regressor",np.round(np.mean(scores),2))
     

