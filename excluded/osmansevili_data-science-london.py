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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


train=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
test=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
train_labels=pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)


train.head()


test.head()


train_labels.head()


train.columns=['col_'+str(i) for i in range(1,train.shape[1]+1)]
test.columns=['col_'+str(i) for i in range(1,train.shape[1]+1)]
train['label']=train_labels


train.head()


train.info()


test.info()


train.describe()


plt.figure(figsize=(10,8))
plt.grid(True)
sns.countplot(data=train,x='label')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.violinplot(data=train,x='label',y='col_13')


plt.figure(figsize=(10,8))
plt.grid(True)
sns.scatterplot(data=train, x='col_13',y='col_29', hue='label')


from sklearn.ensemble import RandomForestClassifier


X=train.drop('label',axis=1)
y=train['label']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)


errors = list()
for i in range(50,501,50):
    clf = RandomForestClassifier(n_estimators=i, max_depth=5, random_state=42)
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    errors.append(np.mean(pred!=y_test))


clf = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=42)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)


print(confusion_matrix(y_test,pred),'\n',classification_report(y_test,pred))


pred=clf.predict(test)

submission = pd.DataFrame({
    "Id": range(1, len(pred)+1),
    "Solution": pred
})

submission.to_csv("submission.csv", index=False)


submission




