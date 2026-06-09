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


#Importing useful packages
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import statistics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
import datetime as dt


train_file_path = '/kaggle/input/playground-series-s5e1/train.csv'
train = pd.read_csv(train_file_path)

train['num_sold'] = train['num_sold'].fillna(value = 0)
train.info()


train['date'] = pd.to_datetime(train['date'])


train['month'] = train['date'].dt.month


train.head()


y = train.num_sold
features = ['country','store','product','month']
X = train[features]
X = pd.get_dummies(X)
X.head()


X.columns


feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)


from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print('The accuracy of the model is: ', rf_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', rf_model.score(train_X, train_y))


rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X,y)


test_data_path = '/kaggle/input/playground-series-s5e1/test.csv'
test = pd.read_csv(test_data_path)
test['date'] = pd.to_datetime(test['date'])
test['month'] = test['date'].dt.month


test_X = test[features]
test_X = pd.get_dummies(test_X)


test_preds = rf_model_on_full_data.predict(test_X)


output = pd.DataFrame({'id': test.id,
                       'num_sales': test_preds})
output.to_csv('submission.csv', index=False)


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

