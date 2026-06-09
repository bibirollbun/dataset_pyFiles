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


submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


submission.head()


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test_data.head()


train_data.isnull().sum()


test_data.isnull().sum()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


x = train_data.drop(columns=["id", "rainfall"])  # Drop 'id' and target column
y = train_data["rainfall"]  # Target variable


x


y


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train


lr = LogisticRegression(max_iter=1000)


lr.fit(x_train,y_train)


y_pred = lr.predict(x_test)


y_pred


accuracy = accuracy_score(y_test,y_pred)


accuracy


test_data.shape


test_data


x_test = test_data.drop(columns=["id"])


x_test


x_test.isnull().sum()


# Fill missing values with the median
x_test.fillna(x_test.median(), inplace=True)


x_test.isnull().sum()


x_test.shape


test_pred = lr.predict(x_test)


submission = pd.DataFrame({"id": test_data["id"], "rainfall": test_pred})


submission.to_csv("submission_new.csv")


sub = pd.read_csv("/kaggle/working/submission_new.csv")


sub.shape

