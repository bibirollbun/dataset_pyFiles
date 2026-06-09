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


%load_ext cudf.pandas
import pandas as pd


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


%%time
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')


%%time
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


print('train shape: ', train.shape)
print('test shape: ', test.shape)


print('train nulls: ', train.isnull().sum())
print('test nulls: ', test.isnull().sum())


train.info()


test.info()


combined = pd.concat([train, test], axis = 0, ignore_index = True)


test.info()


untch = ['id', 'Price']
features = [c for c in train.columns if not c in untch]

cats = []
print(f"The {len(features)} Basic Features Are: ")

for c in features:
    ftype = 'numerical'
    if combined[c].dtype == 'object':
        cats.append(c)
        combined[c].fillna('NAN')
        combined[c], _ = combined[c].factorize()
        combined[c] -= combined[c].min()
        ftype = 'categorical'
    if combined[c].dtype == 'int64':
        combined[c] = combined[c].astype('int32')
    if combined[c].dtype == 'float64':
        combined[c] = combined[c].astype('float32')

    n = combined[c].nunique()
    print(f'{c} have {n} unique values with {ftype}')

train = combined.iloc[:len(train), :].copy()
test = combined.iloc[len(train):, :].reset_index(drop=True).copy()



test.info()


print('train nulls: ', train.isnull().sum())
print('test nulls: ', test.isnull().sum())


train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean()) 


test['Weight Capacity (kg)'] = test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean())


print('train nulls: \n', train.isnull().sum())
print('test nulls: \n', test.isnull().sum())


x, y = train.iloc[:, :-1], train.iloc[:, -1]


x = x.set_index('id')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(
    objective="reg:squarederror",  # For regression tasks
    n_estimators=100,  # Number of trees (increase for better accuracy)
    learning_rate=0.1,  # Step size (reduce if overfitting)
    max_depth=6,  # Tree depth (increase if needed)
    subsample=0.8,  # Use 80% of data for each tree
    colsample_bytree=0.8,  # Use 80% of features for each tree
    random_state=42
)


model.fit(x_train, y_train)


y_pred = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse:.5f}")
print(f"Root Mean Squared Error: {rmse:.5f}")


train_x = train.drop(columns=['Price'])
train_y = train['Price']


print('train x shape: ', train_x.shape)
print('train y shape: ', train_y.shape)


model.fit(train_x, train_y)


train.info()


test.info()


test = test.drop(columns = ['Price'])


y_pred_final = model.predict(test)


test_y = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


test_y['Price'] = y_pred_final


test_y


test_y.to_csv('submission.csv', index=False)




