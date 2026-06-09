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
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_data.head()


train_data.shape[0]


print(train_data['Brand'].unique())
print(train_data['Material'].unique())
print(train_data['Size'].unique())
print(train_data['Compartments'].unique())
print(train_data['Laptop Compartment'].unique())
print(train_data['Waterproof'].unique())
print(train_data['Style'].unique())
print(train_data['Color'].unique())
print(train_data['Weight Capacity (kg)'].unique())
print(train_data['Price'].unique())


graph_data = pd.pivot_table(train_data, index='Brand', columns='Style', values='Price', aggfunc='sum')
graph_data.head()


plt.hist
plt.plot(list(graph_data.index), graph_data['Backpack'], label='Backpack')
plt.plot(list(graph_data.index), graph_data['Messenger'], label='Messenger')
plt.plot(list(graph_data.index), graph_data['Tote'], label='Tote')
plt.legend()


train_data.isnull().any(axis=0)


train_data['Brand'] = train_data['Brand'].fillna('Missing')
train_data['Material'] = train_data['Material'].fillna('Missing')
train_data['Size'] = train_data['Size'].fillna('Missing')
train_data['Laptop Compartment'] = train_data['Laptop Compartment'].fillna('Missing')
train_data['Waterproof'] = train_data['Waterproof'].fillna('Missing')
train_data['Style'] = train_data['Style'].fillna('Missing')
train_data['Color'] = train_data['Color'].fillna('Missing')
train_data['Weight Capacity (kg)'] = train_data['Weight Capacity (kg)'].fillna(train_data.mean(numeric_only=True))


list_cols = ['Brand','Material','Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='ignore')
train_data_ce_onehot = ce_ohe.fit_transform(train_data)

train_data_ce_onehot.head()


X = train_data_ce_onehot[['Brand_1', 'Brand_2', 'Brand_3', 'Brand_4', 'Brand_5', 'Brand_6',
        'Material_1', 'Material_2', 'Material_3', 'Material_4', 'Material_5', 'Style_1',
        'Style_2', 'Style_3', 'Style_4', 'Color_1', 'Color_2', 'Color_3', 'Color_4',
         'Weight Capacity (kg)']]
Y = train_data_ce_onehot['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_test = lgb.Dataset(X_test, Y_test, reference=lgb_train)

model = lgb.train(params, lgb_train, valid_sets=[lgb_test], valid_names=["valid"], num_boost_round=100)


pred_test = model.predict(X_test)
mse = mean_squared_error(Y_test, pred_test)
print(f'Mean Squared Error (MSE): {mse}')


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_data.head()


test_data['Brand'] = test_data['Brand'].fillna('Missing')
test_data['Material'] = test_data['Material'].fillna('Missing')
test_data['Size'] = test_data['Size'].fillna('Missing')
test_data['Laptop Compartment'] = test_data['Laptop Compartment'].fillna('Missing')
test_data['Waterproof'] = test_data['Waterproof'].fillna('Missing')
test_data['Style'] = test_data['Style'].fillna('Missing')
test_data['Color'] = test_data['Color'].fillna('Missing')
test_data['Weight Capacity (kg)'] = test_data['Weight Capacity (kg)'].fillna(test_data.mean(numeric_only=True))


list_cols = ['Brand','Material','Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='ignore')
test_data_ce_onehot = ce_ohe.fit_transform(test_data)

test_data_ce_onehot.head()


X_2 = test_data_ce_onehot[['Brand_1', 'Brand_2', 'Brand_3', 'Brand_4', 'Brand_5', 'Brand_6',
        'Material_1', 'Material_2', 'Material_3', 'Material_4', 'Material_5', 'Style_1',
        'Style_2', 'Style_3', 'Style_4', 'Color_1', 'Color_2', 'Color_3', 'Color_4',
         'Weight Capacity (kg)']]


new_predictions = model.predict(X_2)
print(f'Prediction results for new data: {new_predictions}')


len(test_data)


len(new_predictions)


result_data = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
result_data.head()


result_data['Price'] = new_predictions
result_data.head()


len(result_data)


result_data.to_csv('/kaggle/working/output.csv', index=False)

