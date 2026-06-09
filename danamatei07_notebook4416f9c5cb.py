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

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


num_cols = ['Compartments', 'Weight Capacity (kg)']
train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(train[num_cols].median())

cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
train[cat_cols] = train[cat_cols].fillna('Unknown')
test[cat_cols] = test[cat_cols].fillna('Unknown')


train['is_large_capacity'] = (train['Weight Capacity (kg)'] > 20).astype(int)
test['is_large_capacity'] = (test['Weight Capacity (kg)'] > 20).astype(int)


color_counts = train['Color'].value_counts().to_dict()
train['color_freq'] = train['Color'].map(color_counts)
test['color_freq'] = test['Color'].map(color_counts).fillna(0)

target_col = 'Price'
X = train.drop(columns=['id', target_col])
y = train[target_col]
test_data = test.drop(columns=['id'])


X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)
X, test_data = X.align(test_data, join='left', axis=1, fill_value=0)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("RMSE on validation set:", rmse)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 wichtigsten Features")
plt.gca().invert_yaxis()
plt.show()


final_preds = model.predict(test_data)


submission = pd.DataFrame({
    'id': test['id'],
    'Price': final_preds
})
submission.to_csv('submission.csv', index=False)

