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


full_train = pd.read_csv("/kaggle/input/pwaic-iris-dataset-competition/training.csv")
full_train.head()


test = pd.read_csv("/kaggle/input/pwaic-iris-dataset-competition/testing.csv")
test.head()


y = full_train['label']
train = full_train.drop(['label'], axis=1)


full_train.info()
print()
test.info()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train, y)
y_train = model.predict(train)
y_train


from sklearn.metrics import f1_score, accuracy_score
print("f1_score", f1_score(y, y_train, average=None))
print("accuracy", accuracy_score(y, y_train))


y_pred = model.predict(test)
output = pd.DataFrame({'id':range(1,len(y_pred)+1),'label':y_pred})
output.to_csv('submission.csv', index=False)




