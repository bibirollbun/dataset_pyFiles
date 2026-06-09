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


train_path="/kaggle/input/playground-series-s5e2/train.csv"
train_data=pd.read_csv(train_path)
train_data.head()


test_path="/kaggle/input/playground-series-s5e2/test.csv"
test_data=pd.read_csv(test_path)
test_data


y=train_data[["Price"]]
X=train_data[["Compartments","Weight Capacity (kg)"]]
X


import xgboost as xgb
model=xgb.XGBRFRegressor()
model.fit(X,y)


test_data[test_data.columns[2:]]


y_pred=model.predict(test_data[["Compartments","Weight Capacity (kg)"]])
print(y_pred)
submission = pd.DataFrame(y_pred,columns=[["Price"]])
submission.insert(0, 'id', test_data['id']) 
print(submission)


submission.to_csv('submission.csv', index=False)

