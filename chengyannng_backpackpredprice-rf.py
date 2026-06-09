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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.manifold import TSNE


data = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data.info()


categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
missing_categorical = data[categorical_features].isnull().any(axis=1)
data = data[~missing_categorical]
data.info()
data.head()


data = data.drop(columns=['id'])


data_encoded = pd.get_dummies(data, drop_first=True)
bool_cols = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_cols] = data_encoded[bool_cols].astype(int)


data_encoded = data_encoded.sample(n=50000, random_state=42)
data_encoded = data_encoded.reset_index(drop=True)
data_encoded


X_NoPrice = data_encoded.drop(columns='Price')
y = data_encoded['Price']


X_train, X_test, y_train, y_test = train_test_split(X_NoPrice, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("result：")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


test_data = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

test_ids = test_data['id'] 
test_data.drop(columns=['id'], inplace=True)

for col in test_data.columns:
    if test_data[col].dtype == "object":
        test_data[col].fillna("None", inplace=True)
    else:
        test_data[col].fillna(test_data[col].mean(), inplace=True)

test_data_encoded = pd.get_dummies(test_data, drop_first=True)
bool_cols = test_data_encoded.select_dtypes(include=['bool']).columns
test_data_encoded[bool_cols] = test_data_encoded[bool_cols].astype(int)

missing_cols = set(X_NoPrice.columns) - set(test_data_encoded.columns)
for col in missing_cols:
    test_data_encoded[col] = 0

test_data_encoded = test_data_encoded[X_NoPrice.columns]



test_predictions = model.predict(test_data_encoded)

submission = pd.DataFrame({"id": test_ids, "Price": test_predictions})
submission.set_index("id", inplace=True)

print(submission)

submission.to_csv('BackpackPricePred_RF.csv')

