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


train_file_path = "/kaggle/input/playground-series-s5e2/train.csv"
train_data = pd.read_csv(train_file_path)
train_data.describe()


train_data.columns


train_data_cleaned = train_data.dropna(axis=0)

features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof',
    'Style', 'Color', 'Compartments', 'Weight Capacity (kg)']

y = train_data_cleaned.Price
X = train_data_cleaned[features]

X_encoded = pd.get_dummies(X)
X_encoded.describe()


X.head()


from sklearn.tree import DecisionTreeRegressor

backpack_model = DecisionTreeRegressor(random_state=1)

backpack_model.fit(X_encoded, y)


print("Making predictions for the following 5 backpacks:")
print(X_encoded.head())
print("The predictions are")
print(backpack_model.predict(X_encoded.head()))
print("The actual prices are")
print(y.head())


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_X, val_X, train_y, val_y = train_test_split(X_encoded, y, random_state = 0)
backpack_model = DecisionTreeRegressor()
backpack_model.fit(train_X, train_y)

val_predictions = backpack_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_data_cleaned = test_data.dropna(axis=0)
X_test = test_data_cleaned[features]
X_test_encoded = pd.get_dummies(X_test)

X_test_encoded = X_test_encoded.reindex(columns = X_encoded.columns, fill_value=0)

test_preds = backpack_model.predict(X_test_encoded)
submission = pd.DataFrame({'id': test_data_cleaned['id'], 'Price': test_preds})
submission.to_csv('submission.csv', index=False)

