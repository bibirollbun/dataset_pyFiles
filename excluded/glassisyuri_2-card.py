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


## 평가 지표 : roc-auc
## y : Class

import pandas as pd
train = pd.read_csv("/kaggle/input/big-data-analytics-certification-kr-2024-2/train.csv")
test = pd.read_csv("/kaggle/input/big-data-analytics-certification-kr-2024-2/test.csv")



print(train.shape, test.shape)


print(train.info())


print(train.isnull().sum())
print("\n")
print(test.isnull().sum())


train.drop('id', axis=1, inplace=True)


target = train.pop("Class")
print(train.shape, test.shape)



from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train, target, test_size=0.2, random_state=0
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict_proba(X_val)


from sklearn.metrics import roc_auc_score
result = roc_auc_score(y_val, pred[:,1])
print(result)


test_id = test.pop('id')


res = rf.predict_proba(test)[:,1]
df = pd.DataFrame({'id' : test_id, 'Class' : res})
df.to_csv('result.csv', index = False)


pd.read_csv("/kaggle/working/result.csv")

