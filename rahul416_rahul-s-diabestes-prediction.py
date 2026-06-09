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


import numpy as np
import pandas as pd
train =  pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


train.info()


test.info()


train.columns


cols_to_remove = ['gender', 'ethnicity', 'education_level','income_level', 'smoking_status', 'employment_status']
train.drop(cols_to_remove,axis=1,inplace=True)
test.drop(cols_to_remove,axis=1,inplace=True)



id_ = test['id']
train.drop(['id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)


test_df = test


from sklearn.preprocessing import MinMaxScaler

X = train.drop(['diagnosed_diabetes'],axis=1)
y = train['diagnosed_diabetes']

scaler = MinMaxScaler()

X_ = scaler.fit_transform(X)
test = scaler.transform(test)


from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(X_,y,test_size=0.2)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=10)

model.fit(x_train,y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))


# from sklearn import tree
# import matplotlib.pyplot as plt
# tree.plot_tree(model)


y_pred = model.predict(test)
print(y_pred)











X = train.drop(['diagnosed_diabetes'],axis=1)
y = train['diagnosed_diabetes']



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)
test_ = norm.transform(test_df)


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = DecisionTreeClassifier(criterion='entropy')

estimators = [("Logistic",model1),("DT-gini",model2),("DT-entropy",model3)]

vc = VotingClassifier(estimators=estimators,voting='soft')
vc.fit(X_train,y_train)


from sklearn.metrics import classification_report
y_test_pred = vc.predict(X_test)
print(classification_report(y_test,y_test_pred))


from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression


model1 = LogisticRegression()
bag = BaggingClassifier(estimator=model1,
                       n_estimators=100,
                       max_samples = 0.10)
bag.fit(X_train,y_train)


from sklearn.metrics import classification_report
y_test_pred = bag.predict(X_test)
print(classification_report(y_test,y_test_pred))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,   
                            criterion='gini',   # gini entropy
                            max_depth=None,
                            oob_score=True,   # out of bag evaluation --> cross validation
                            
              
                            )
rf.fit(X_train,y_train)


from sklearn.metrics import classification_report
y_test_pred = rf.predict(X_test)
print(classification_report(y_test,y_test_pred))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
m1 = DecisionTreeClassifier()

ada = AdaBoostClassifier(
                         estimator=DecisionTreeClassifier(),
                         n_estimators=10,
                         learning_rate=0.1,
                         )
ada.fit(X_train,y_train)




from sklearn.metrics import classification_report
y_pred = ada.predict(X_test)
print(classification_report(y_test,y_pred))
#


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier( 
                                    loss = 'log_loss', # 'exponential'
                                    n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=1)
gb.fit(X_train,y_train)




from sklearn.metrics import classification_report
y_pred = gb.predict(X_test)
print(classification_report(y_test,y_pred))





from xgboost import XGBClassifier

neg, pos = np.bincount(y_train)

scale_pos_weight = neg / pos

xgb = XGBClassifier(
                        # objective = '',
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=1,
                        scale_pos_weight=scale_pos_weight,
                      )
xgb.fit(X_train,y_train)


from sklearn.metrics import classification_report
y_pred = xgb.predict(X_test)
print(classification_report(y_test,y_pred))


y1 = ada.predict(test_)
y2 = gb.predict(test_)
y3 = xgb.predict(test_)
y4 = rf.predict(test_)
y5 = bag.predict(test_)

y_pred = []
from statistics import mode
for i in range(len(y1)):
    yi = mode((y4[i],y5[i],y3[i],y3[i],y3[i],y2[i],y1[i]))
    y_pred.append(int(yi))
# print(y_pred)




df = pd.DataFrame(
    {
    'id':id_,
    'diagnosed_diabetes':y_pred
    }
)






df.to_csv('submission.csv',index=False)


df




