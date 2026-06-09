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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


submission.head()


train.head()


train.info()


train.shape


train.isnull().sum()


train = train.dropna()


test.info()


test=test.dropna()


train['Stage_fear'].value_counts()


train['Drained_after_socializing'].value_counts()


train['Personality'].value_counts()


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


train.head()


test_ids = test['id'].copy()


x_train=train.drop(['Personality'],axis=1)
y_train=train[['Personality']]

x_test = test.copy()


x_train = x_train.drop(['id'], axis=1)
x_test = x_test.drop(['id'], axis=1)


x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(x_test, drop_first=True)


x_train, x_test, y_train, y_test=train_test_split(x_train, y_train, test_size=0.20, random_state=42)


g=GaussianNB()
b=BernoulliNB()


g.fit(x_train,y_train)


b.fit(x_train,y_train)


gtahmin=g.predict(x_test)


accuracy_score(y_test, gtahmin)


btahmin=b.predict(x_test)


accuracy_score(y_test, btahmin)


confusion_matrix(y_test, btahmin)


confusion_matrix(y_test, gtahmin)


sns.heatmap(confusion_matrix(y_test, gtahmin), annot=True)


print(classification_report(y_test,gtahmin))
print(classification_report(y_test,btahmin))




