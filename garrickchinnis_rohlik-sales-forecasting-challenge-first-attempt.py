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


# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


sales_train_file_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv'
sales_train = pd.read_csv(sales_train_file_path).dropna()


sales_train.head()


y =sales_train.sales
features = ['warehouse','total_orders','sell_price_main']
X = sales_train[features]
X = pd.get_dummies(X)



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



print('The accuracy of the model is: ', rf_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', rf_model.score(train_X, train_y))


rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X,y)
sales_test_path = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv'
sales_test = pd.read_csv(sales_test_path)
test_X = sales_test[features]
test_X = pd.get_dummies(test_X)
test_preds = rf_model_on_full_data.predict(test_X)


id = sales_test.unique_id + _ + sales_test.date


output = pd.DataFrame({'id': sales_test.unique_id,
                       'Sales': test_preds})
output.to_csv('submission.csv', index=False)

