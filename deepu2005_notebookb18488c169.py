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


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#load dataset
data = load_wine()
X = data.data
Y = data.target


print(X)


print(Y)


winedf = pd.DataFrame(data.data, columns=data.feature_names)
winedf


#split wine data into train set and test set
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.3,random_state = 42)


model = LogisticRegression(max_iter=200)
model.fit(X_train,Y_train)


#evalute th emodel
Y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,Y_pred))
# print(Y_pred)

