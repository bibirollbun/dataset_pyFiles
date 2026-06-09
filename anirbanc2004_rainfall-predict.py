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


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


train.info()


train.describe()


variances = train.var().sort_values(ascending=False)
print(f'The variances of all the features are in descending order : \n{variances}')


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


sc = StandardScaler()


X = train.drop(columns=['id','day','temparature','rainfall'],inplace=False)
Y = train['rainfall']


X = sc.fit_transform(X)


X_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


model_rf = RandomForestClassifier(n_estimators=500,max_depth=12,random_state=42)


model_rf.fit(X_train,y_train)


model_rf.score(X_train,y_train)


predict_rf = model_rf.predict(x_test)


from sklearn.metrics import roc_auc_score

roc_val = roc_auc_score(predict_rf, y_test)
print(roc_val)


test


X_test = test.drop(columns=['id','day','temparature'],inplace=False)
# Y_test = test['rainfall']


X_test.info()


X_test_final = X_test.fillna(method='ffill')


X_test_final.info()


predict_submission = model_rf.predict(X_test_final)


output = pd.DataFrame({'id': test.id, 'rainfall':predict_submission})
output.to_csv('Submission1.csv' , index=False)
print("Your submission is saved")

