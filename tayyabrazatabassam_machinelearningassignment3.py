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
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"/kaggle/input/machinelearningassignment3/train.csv")


df.dropna(axis=0,inplace= True)


x = df.iloc[:, :-1]
y = df ['label']


scaller = StandardScaler()
x = scaller.fit_transform(x)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)


from sklearn.model_selection import train_test_split

import xgboost as xgb


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)



XGB = xgb.XGBClassifier()
XGB.fit(x_train,y_train)
XGB_pred = XGB.predict(x_test)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test,XGB_pred)
classification_report = classification_report(y_test,XGB_pred)
confusion_matrix = confusion_matrix(y_test,XGB_pred)
print(f"Accuracy: {accuracy}")



print(f"classification_report: {classification_report}")
print(f"confusion_matrix: {confusion_matrix}")

