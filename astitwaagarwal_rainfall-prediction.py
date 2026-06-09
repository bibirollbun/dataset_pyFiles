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


import pandas as pd, numpy as np
import warnings
warnings.simplefilter('ignore')
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col='id')
train_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")


train_extra.columns = train_extra.columns.str.replace(' ', '')
train_extra = train_extra[train_extra.columns].copy()
train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
train_extra['humidity']=train_extra['humidity'].astype(float)
train_extra['cloud']=train_extra['cloud'].astype(float)
train_features=list(train)
train_extra=train_extra[train_features]


train = pd.concat([train, train_extra], axis=0, ignore_index=True)
train = train.drop_duplicates()
train.shape


train['cloud'].min(),train['cloud'].max(),test['cloud'].min(),test['cloud'].max()


train['sunshine'].min(),train['sunshine'].max(),test['sunshine'].min(),test['sunshine'].max()


train['day'].min(),train['day'].max(),test['day'].min(),test['day'].max()


features=list(test)
features.append('rainfall')
train=train[features]


target = "rainfall"


test['winddirection']=test['winddirection'].fillna(value=test['winddirection'].mean())
train['winddirection']=train['winddirection'].fillna(value=train['winddirection'].mean())
train['windspeed']=train['windspeed'].fillna(value=train['windspeed'].mean()) 


class_weight={0:657.0/(1899+657), 1:1899.0/(1899+657)}
class_weight


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y=train['rainfall']
drop_features=['cloud','humidity']
drop_features.append('rainfall')
X=train.drop(columns=drop_features,axis=1)
print(X.head(2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf_1 = LogisticRegression(solver='liblinear',penalty='l1',max_iter=10000, random_state=88,C=1.0) #81.05
clf_1.fit(X_train, y_train)
y_pred_1 =  clf_1.predict(X_test)   
acc = accuracy_score(y_test,y_pred_1 ) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


drop_features.remove('rainfall')
_test=test.drop(columns=drop_features,axis=1)
test_preds_1 = clf_1.predict_proba(_test)[:,1]


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y=train['rainfall']
drop_features=['day','mintemp','pressure','sunshine','winddirection','windspeed','maxtemp','dewpoint','temparature']
drop_features.append('rainfall')
X=train.drop(columns=drop_features,axis=1)
print(X.head(2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf_2 = LogisticRegression(solver='newton-cg',penalty='none',max_iter=10000, random_state=43,C=1.0) #83.79
#clf_2 = LogisticRegression(solver='saga',penalty='l2',max_iter=10000, random_state=42,C=1.0) #83.20
clf_2.fit(X_train, y_train)
    
y_pred_2 =  clf_2.predict(X_test)   
acc = accuracy_score(y_test,y_pred_2 ) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")


drop_features.remove('rainfall')
_test=test.drop(columns=drop_features,axis=1)
test_preds_2 = clf_2.predict_proba(_test)[:,1]


sub = pd.DataFrame({"id": test.index, "rainfall": list(test_preds_1)})
#sub.to_csv("submission.csv", index=False)
sub.head()


sub = pd.DataFrame({"id": test.index, "rainfall": list(test_preds_2)})
#sub.to_csv("submission.csv", index=False)
sub.head()


#sub = pd.read_csv("/kaggle/input/ps-s5e3-rainfall-ensemble-of-solutions/submission.csv")
sub = pd.read_csv("/kaggle/input/ps-s5e3-rainfall-division-attention/submission.csv")


sub['rainfall'] = 0.98 * sub['rainfall'] + 0.02 * test_preds_2 

sub.to_csv("submission.csv", index=False)
sub.head(10)




