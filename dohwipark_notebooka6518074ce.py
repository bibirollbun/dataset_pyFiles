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


import sys
print(sys.version)


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()


train.columns


train.describe()


train['Survived'].value_counts()


train.groupby('Sex')['Survived'].mean()


train.pivot_table('Survived', index='Sex', columns='Pclass')


Age = pd.cut(train['Age'], [0, 18, 80])
train.pivot_table('Survived', ['Sex', Age], 'Pclass')


train.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x = 'Sex', hue = "Survived", data = train)
plt.legend(loc = "upper right", title = "Survived ~ Sex")


sns.stripplot(x='Sex', y='Age', data=train, hue='Survived')


sns.countplot(x = 'SibSp', hue = "Survived", data = train)
plt.legend(loc = "upper right", title = "Survived ~ Sibsp")


sns.countplot(x = 'Parch', hue = "Survived", data = train)
plt.legend(loc = "upper right", title = "Survived ~ Parch")


train['Alone']=np.where((train["SibSp"]+train["Parch"])>0,0,1)
train.drop(['SibSp', 'Parch'], axis=1, inplace=True)


sns.countplot(x = 'Embarked', hue = "Survived", data = train)
plt.legend(loc = "upper right", title = "Survived ~ Embarked")


train.drop('Embarked', axis=1, inplace=True)


train['Ticket']


train.drop('Ticket', axis=1, inplace=True)


train.columns


train.drop(['PassengerId','Name','Cabin'], axis=1, inplace=True)


train


train.corr()


numeric_train = train.select_dtypes(include=np.number)
numeric_train.corr()


train.columns


trainMedian = train["Age"].median(skipna=True)
train["Age"].fillna(trainMedian, inplace=True)


train['Pclass'].value_counts()


train['Sex'].value_counts()


training=pd.get_dummies(train, columns=["Pclass","Sex"], drop_first=True)
training


from sklearn.preprocessing import StandardScaler
train_standard = StandardScaler()
train_copied = training.copy()
train_standard.fit(train_copied[['Age', 'Fare']])
train_std = pd.DataFrame(train_standard.transform(train_copied[['Age', 'Fare']]))
train_std


training[['Age','Fare'] ] = train_std
training


from sklearn.linear_model import LogisticRegression

cols = ["Age","Fare","Alone","Pclass_2","Pclass_3","Sex_male"] 
X =training[cols]
y = training['Survived']

model = LogisticRegression()
model.fit(X,y)


from sklearn.metrics import accuracy_score
train_predicted =model.predict(X)
accuracy_score( train_predicted, y)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(bootstrap=True,n_estimators=10,criterion='gini',max_depth=None,random_state=1)
model.fit(X,y)


predicted_train2 = model.predict(X)
accuracy_score(predicted_train2 , y)


test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()


test.drop(['PassengerId','Name','Cabin','Ticket'], axis=1, inplace=True)


TrainMedian = train['Age'].median()
TrainMedian


test["Age"].fillna(TrainMedian, inplace=True)
test["Fare"].fillna(train.Fare.median(), inplace=True)
test['Alone']=np.where((test["SibSp"]+test["Parch"])>0, 0, 1)
test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
testing=pd.get_dummies(test, columns=["Pclass","Sex"], drop_first=True)
testing


test_copied = testing.copy()
test_std = train_standard.transform(test_copied[['Age','Fare']])

testing[['Age','Fare']] = test_std
testing


X_test=testing[cols]
test_predicted = model.predict(X_test)


test_predicted


sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = list(map(int, test_predicted))
sub.to_csv('submission.csv', index=False)

