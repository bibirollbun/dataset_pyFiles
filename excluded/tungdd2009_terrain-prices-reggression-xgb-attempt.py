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


data = pd.read_csv('/kaggle/input/terrain-prices-reggression/train.csv')


data = pd.get_dummies(data)


data.columns



cols = data.columns
cols = cols.drop(['target'])


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data[cols], data.target, test_size = 0.2, random_state = 42)


y_train.describe()


from xgboost import XGBRegressor


#help(XGBRegressor)


from sklearn.metrics import r2_score



model = XGBRegressor(
    n_estimators = 1000,
    max_depth = 6,
    learning_rate = 0.05,
    colsample_bytree = 0.7,
    subsample = 0.7,
    min_child_weight = 1,
    reg_lambda = 1,
    random_state = 42,
    gamma = 0.3,
    #device = 'cuda',
    eval_metric = ['rmse', 'mae'],
)


model.fit(X_train, y_train, eval_set = [(X_val, y_val)],
         early_stopping_rounds = 50, verbose = 50)


test = pd.read_csv("/kaggle/input/terrain-prices-reggression/test.csv")


# On test data, get dummies
dtest = pd.get_dummies(test)

# Reindex test columns to train dummy columns, fill missing cols with 0
dtest = dtest.reindex(columns=cols, fill_value=0)

#variables = variables.drop('SalePrice')
y_test = model.predict(dtest[cols])

print(len(test))          # probably 1460
print(len(y_test))        # probably 



test.head()



y_testd = pd.DataFrame({
    'id': test['id'],
    'target': y_test
})
y_testd.to_csv('predictions.csv', index=0)


y_testd




