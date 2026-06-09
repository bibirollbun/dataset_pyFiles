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


train_ds = pd.read_csv("/kaggle/input/tabular-playground-series-mar-2021/train.csv")
test_ds = pd.read_csv("/kaggle/input/tabular-playground-series-mar-2021/test.csv")


train_ds


train_ds.head()


from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()


objects_train = train_ds.select_dtypes(include = "object")
objects_test = test_ds.select_dtypes(include = "object")


for objects in objects_train:
    train_ds[objects]  = encoder.fit_transform(train_ds[objects])

for objects in objects_test:
    test_ds[objects]  = encoder.fit_transform(test_ds[objects])


train_ds.head()


train_X  = train_ds.drop(["id", "target"], axis = 1)
train_y = train_ds["target"]

test_X = test_ds.drop("id", axis = 1)


from sklearn.linear_model import LogisticRegression


model = LogisticRegression()


model.fit(train_X, train_y)


y_preds= model.predict_proba(test_X)


y_preds


rounded_probs = np.round(y_preds, 1)[:, 1]
rounded_probs


prediction = []


for arrays in rounded_probs:
    prediction.append(arrays)


prediction


submission = pd.DataFrame({
    "id" : test_ds["id"],
    "target": rounded_probs
})


submission.to_csv("tabular.csv", index = False)




