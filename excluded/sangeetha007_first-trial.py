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


train=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train.head()


print(train.isnull())


print(train.isna().any())
#the train dataset is clean


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print(test.head())
print(test.isna().any())


from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print(train.columns)
print(test.columns)





X_train = train.drop("diagnosed_diabetes", axis=1)
y_train = train["diagnosed_diabetes"]

#X_test = test.drop("diagnosed_diabetes", axis=1)
#y_test = test["diagnosed_diabetes"]


cat_cols = ["gender", "ethnicity", "education_level", "income_level",
            "smoking_status", "employment_status"]

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")


