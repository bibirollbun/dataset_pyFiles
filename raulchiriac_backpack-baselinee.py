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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')



print(train.head())
print(train.info())



X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_test = test.drop(columns=['id'])

combined = pd.concat([X, X_test], axis=0)

combined_encoded = pd.get_dummies(combined)

combined_encoded = combined_encoded.fillna(combined_encoded.mean(numeric_only=True))

X = combined_encoded.iloc[:len(X), :]
X_test = combined_encoded.iloc[len(X):, :]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE:", rmse)



preds = model.predict(X_test)
submission['Price'] = preds
submission.to_csv('submission.csv', index=False)
submission.head()


