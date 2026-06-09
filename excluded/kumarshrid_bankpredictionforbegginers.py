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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


#CHECK FOR NULL VALUES
train_data.isnull().sum()


train_data.drop('id',axis = 1,inplace = True)
pap = test_data['id']
test_data.drop('id',axis = 1,inplace = True)


train_data['contact'].value_counts()
#DO ONE HOT ENCODING
train_ = pd.get_dummies(train_data)
test_ = pd.get_dummies(test_data)


#CORRELATION WITH Y FEAUTURE
train_.corr()['y']


#TRAINING THE MODEL WITH LGBM AND PARAMETERS
from xgboost import XGBClassifier
y = train_['y']
train_.drop('y',axis = 1,inplace = True)
X = train_
model = XGBClassifier(
            n_estimators=1582,  # Più alberi
    learning_rate= 0.055238410897498764,  # Più lento = più robusto
    max_depth=6,
    subsample=0.9464704583099741,
    colsample_bytree= 0.6624074561769746,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
        )
model.fit(X,y)


oupu = model.predict(test_)


output = pd.DataFrame({'id' : pap,'y' : oupu})
output.to_csv('submission.csv',index = False)

