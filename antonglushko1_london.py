import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


train= pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test= pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)

labels= pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)


print('Training Data: ', train.shape)
print('Testing Data: ', test.shape)
print('Labels: ', labels.shape)


train.columns


train.head()


labels.head()


lg= LogisticRegression()
lg.fit(train, labels.values.ravel())
lg_out= lg.predict(test)


dtc= DecisionTreeClassifier()
dtc.fit(train, labels.values.ravel())
dtc_out= dtc.predict(test)


print('LogisticRegression and DecisionTreeClassifier: ', accuracy_score(lg_out, dtc_out))


knn= KNeighborsClassifier()
knn.fit(train, labels.values.ravel())
knn_out= knn.predict(test)


print('KNN and LogisticRegression: ', accuracy_score(lg_out, knn_out))
print('KNN and DecisionTreeClassifier: ', accuracy_score(dtc_out, knn_out))


svc= SVC()
svc.fit(train, labels.values.ravel())
svc_out= svc.predict(test)


print('SVC and LogisticRegression',accuracy_score(lg_out, svc_out))
print('SVC and DecisionTreeClassifier',accuracy_score(dtc_out, svc_out))
print('SVC and KNN',accuracy_score(knn_out, svc_out))


gbc= GradientBoostingClassifier(random_state=0)
gbc.fit(train, labels.values.ravel())
gbc_out= gbc.predict(test)


print('GradientBoosting and LogisticRegression',accuracy_score(lg_out, gbc_out))
print('GradientBoosting and DecisionTreeClassifier',accuracy_score(dtc_out, gbc_out))
print('GradientBoosting and KNN',accuracy_score(knn_out, gbc_out))
print('GradientBoosting and SVC',accuracy_score(svc_out, gbc_out))


cbc = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0,loss_function='Logloss', random_seed=42)
cbc.fit(train, labels.values.ravel())
cbc_out = cbc.predict(test)


print('CatBoost and LogisticRegression',accuracy_score(lg_out, cbc_out))
print('CatBoost and DecisionTreeClassifier',accuracy_score(dtc_out, cbc_out))
print('CatBoost and KNN',accuracy_score(knn_out, cbc_out))
print('CatBoost and SVC',accuracy_score(svc_out, cbc_out))
print('CatBoost and GradientBoosting',accuracy_score(gbc_out, cbc_out))


print('SVC and LogisticRegression',accuracy_score(lg_out, svc_out))
print('SVC and DecisionTreeClassifier',accuracy_score(dtc_out, svc_out))
print('SVC and KNN',accuracy_score(knn_out, svc_out))
print('SVC and CatBoost',accuracy_score(svc_out, cbc_out))
print('SVC and GradientBoosting',accuracy_score(gbc_out, svc_out))


svc_out


sol= pd.DataFrame({'Id': range(1, len(svc_out)+1), 'Solution': svc_out})


sol.to_csv('submission.csv', index=False)

