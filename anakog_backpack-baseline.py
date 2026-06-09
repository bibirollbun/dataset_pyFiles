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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

train_data.head()


test_data.head()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train_data['Laptop Compartment'] = train_data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
train_data['Waterproof'] = train_data['Waterproof'].map({'Yes': 1, 'No': 0})
test_data['Laptop Compartment'] = test_data['Laptop Compartment'].map({'Yes': 1, 'No': 0})
test_data['Waterproof'] = test_data['Waterproof'].map({'Yes': 1, 'No': 0})

encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_columns = ['Brand', 'Material', 'Size', 'Style', 'Color']
X_train_cat = encoder.fit_transform(train_data[categorical_columns])
X_test_cat = encoder.transform(test_data[categorical_columns])

scaler = StandardScaler()
numerical_columns = ['Compartments', 'Weight Capacity (kg)']
X_train_num = scaler.fit_transform(train_data[numerical_columns])
X_test_num = scaler.transform(test_data[numerical_columns])

X_train = pd.concat([pd.DataFrame(X_train_cat), pd.DataFrame(X_train_num)], axis=1)
X_test = pd.concat([pd.DataFrame(X_test_cat), pd.DataFrame(X_test_num)], axis=1)

y_train = train_data['Price']



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
submission = pd.DataFrame({'id': test_data['id'], 'Price': y_pred})
submission.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

