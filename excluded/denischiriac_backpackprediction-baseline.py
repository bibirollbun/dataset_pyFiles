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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

train_data.head()


test_data.head()


target_col = 'Price'


X = train.drop(columns=['id', target_col])
y = train[target_col]
X_test = test.drop(columns=['id'])


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)


X_test = X_test.reindex(columns=X.columns, fill_value=0)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f"Baseline Random Forest RMSE: {rmse_rf:.4f}")



final_preds = rf_model.predict(X_test)
submission = sample_submission.copy()
submission['Price'] = final_preds
submission.to_csv('submission.csv', index=False)


