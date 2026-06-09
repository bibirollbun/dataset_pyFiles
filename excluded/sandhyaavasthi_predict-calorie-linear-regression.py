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


from sklearn.linear_model import LinearRegression


# load train and test data
train=pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


#top 5 rows of train data
train.head()


X_train=train[["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]]
X_test=test[["Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]]
y_train=train["Calories"]


#initialize model
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)


print(predictions)


test


submission=test[["id"]]



submission ['Calories']= predictions


submission.head(15)


# to replace negative value with 0
predictions = model.predict(X_test)
predictions = np.maximum(0, predictions)  # set negative predictions to zero


submission ['Calories']= predictions


# save prediction
submission.to_csv("submission.csv", index=None)


submission.tail(15)

