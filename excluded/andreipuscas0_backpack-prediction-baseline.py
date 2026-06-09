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


import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train.head()


train.columns


X = train.drop(columns=["id", "Price"])
y = train["Price"].astype(float)

X_test = test.drop(columns=["id"])


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)


print(type(y))
print(y.shape)
print(y.head())


X_sample = X.sample(n=100000, random_state=42)
y_sample = y[X_sample.index]


print(type(y_sample))
print(y_sample.shape)
print(y_sample[:5])


import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

y_sample = np.array(y_sample).astype(np.float64).reshape(-1,)

imputer = SimpleImputer(strategy="mean")
X_sample = imputer.fit_transform(X_sample)
X_test = imputer.transform(X_test)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)
model.fit(X_sample, y_sample)
predictions = model.predict(X_test)


submission["Price"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

