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
warnings.simplefilter('ignore')

from ydata_profiling import ProfileReport


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


train.head()


train.shape


train.duplicated().sum()


train.isnull().sum()


train.isna().sum()


categorical_features = [col for col in train.columns if train[col].dtypes == 'O']
numeric_features = [col for col in train.columns if train[col].dtypes != 'O']

print("Categorical features: ", categorical_features)
print('Numeric features:', numeric_features)


profile = ProfileReport(train, title="Profiling Report")


X = train.drop(columns=['id','y'],axis=1)
Y = train['y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)

print("Size of X train:",X_train.shape)
print("Size of X test:",X_test.shape)
print("Size of y train:",y_train.shape)
print("Size of y test:",y_test.shape)


def categorical_explore(df,feature):
    print("Total number of unique values:",df[feature].nunique())
    print(df[feature].unique())
    print(df[feature].value_counts())
    print("--"*30)

for col in train.columns:
    if col in categorical_features:
        categorical_explore(train,col)


X_train.replace({'job': {'admin.': 'admin'}}, inplace=True)
X_train['job'].replace('unknown', X_train['job'].mode()[0], inplace=True)
X_train['education'].replace('unknown', X_train['education'].mode()[0], inplace=True)
X_train['contact'].replace('unknown', X_train['contact'].mode()[0], inplace=True)
# X_train['poutcome'].replace('unknown', X_train['poutcome'].mode()[0], inplace=True)

X_test.replace({'job': {'admin.': 'admin'}}, inplace=True)
X_test['job'].replace('unknown', X_test['job'].mode()[0], inplace=True)
X_test['education'].replace('unknown', X_test['education'].mode()[0], inplace=True)
X_test['contact'].replace('unknown', X_test['contact'].mode()[0], inplace=True)


for col in X_train.columns:
    if col in categorical_features:
        categorical_explore(X_train,col)


X_train.head()


from sklearn.preprocessing import StandardScaler,LabelEncoder

label = LabelEncoder()
scaler = StandardScaler()

for col in X_train.columns:
    if col in categorical_features:
        X_train[col] = label.fit_transform(X_train[col])
        X_test[col] = label.transform(X_test[col])


X_train


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Classifier accuracy: {acc:.2f}")





