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
from sklearn.ensemble import RandomForestRegressor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train_data.head()


test_data.head()


sample_submission.head()


print(train_data.columns)


features = [
    "Brand",
    "Material",
    "Size",
    "Compartments",
    "Laptop Compartment",
    "Waterproof",
    "Weight Capacity (kg)"
]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
X = X.fillna(0)
X_test = X_test.fillna(0)
y = train_data["Price"]


model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)


predictions = model.predict(X_test)
output = pd.DataFrame({
    "id": test_data["id"],
    "price": predictions
})

output.to_csv("baseline_submission.csv", index=False)

