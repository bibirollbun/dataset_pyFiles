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


train_data.info()


train_data.describe()


train_data.isnull().sum()


from sklearn.impute import SimpleImputer

categorical_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_columns = ['Compartments', 'Weight Capacity (kg)']

# Imputer for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_columns] = cat_imputer.fit_transform(train_data[categorical_columns])
test_data[categorical_columns] = cat_imputer.transform(test_data[categorical_columns])

# Imputer for numerical columns
num_imputer = SimpleImputer(strategy='median')
train_data[numerical_columns] = num_imputer.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = num_imputer.transform(test_data[numerical_columns])



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')

X_train_cat = encoder.fit_transform(train_data[categorical_columns])
X_test_cat = encoder.transform(test_data[categorical_columns])



from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_data[numerical_columns])
X_test_num = scaler.transform(test_data[numerical_columns])


X_train = pd.concat([pd.DataFrame(X_train_cat), pd.DataFrame(X_train_num)], axis=1)
X_test = pd.concat([pd.DataFrame(X_test_cat), pd.DataFrame(X_test_num)], axis=1)



y_train = train_data["Price"]
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
submission = pd.DataFrame({
    "id": test_data["id"],
    "Price": y_pred
})
submission.to_csv("submission.csv", index=False)


submission.head()

