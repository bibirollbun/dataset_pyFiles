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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


train_df=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train=pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.isna().sum()


test.isna().sum()


train['winddirection'].value_counts


#Dropping id column from test and train as it is not significant here 
train=train.drop(columns=['id'])
test=test.drop(columns=['id'])


test['winddirection']=test['winddirection'].fillna(test['winddirection'].mean())


X=train.drop(columns=['rainfall','day'])
y=train['rainfall']
X_test=pd.DataFrame(test)
X_test=X_test.drop(columns=['day'])
X_train,X_val,Y_train,Y_val=train_test_split(X,y,test_size=0.2, random_state=42)


pca=PCA(n_components=3)
pca.fit(X_train)
x_transform=pca.transform(X_train)
x_transform[0]



sc=StandardScaler()
train_scaled=sc.fit_transform(X_train[['pressure','maxtemp']]).astype('float64')
val_scaled=sc.fit_transform(X_val[['pressure','maxtemp']]).astype('float64')


classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_val)
accuracy= accuracy_score(Y_val, y_pred)
accuracy


classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, Y_train)
y_pred1 = classifier.predict(X_val)
accuracy= accuracy_score(Y_val, y_pred1)
accuracy


test.shape


#considering scaling of the entire dataset and then applying different classification algorithms
X_train[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]=sc.fit_transform(X_train[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]).astype('float64')
X_val[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]=sc.fit_transform(X_val[['pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']]).astype('float64')


#randomforest after scaling
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


y_pred2 = classifier.predict(X_val)
accuracy= accuracy_score(Y_val, y_pred2)
accuracy


#Now let us apply logistic regression 
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, Y_train)

y_pred_log = classifier1.predict_proba(X_test)

# accuracy= accuracy_score(Y_val, y_pred_log)
# accuracy


y_pred_log


classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train, Y_train)

y_pred_svm = classifier1.predict(X_val)
accuracy= accuracy_score(Y_val, y_pred_svm)
accuracy


submission=pd.DataFrame(test_df['id'])
submission['prediction']=y_pred_log
submission

submission.to_csv("submission1.csv",index=False)


submission.to_csv("submission1.csv",index=False)




