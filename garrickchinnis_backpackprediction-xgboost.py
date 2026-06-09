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


#XGBoost 
from xgboost import XGBRegressor, XGBClassifier, plot_importance, plot_tree
#Importing other useful packages
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
from sklearn.preprocessing import StandardScaler


# Load training data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train.info()


train.select_dtypes(include=['object']).columns


train['Brand_num'] = pd.factorize(train['Brand'])[0]
train['Material_num'] = pd.factorize(train['Material'])[0]
train['Size_num'] = pd.factorize(train['Size'])[0]
train['Laptop Compartment_num'] = pd.factorize(train['Laptop Compartment'])[0]
train['Waterproof_num'] = pd.factorize(train['Waterproof'])[0]
train['Style_num'] = pd.factorize(train['Style'])[0]
train['Color_num'] = pd.factorize(train['Color'])[0]



train.select_dtypes(include=['int64','float64']).columns


train['Compartments/Weight'] = train['Compartments']/train['Weight Capacity (kg)']
train['Weight/Compartments'] = train['Weight Capacity (kg)']/train['Compartments']


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns


#Create X,y values
y = train['Price']

features = ['Compartments', 'Weight Capacity (kg)', 'Brand_num',
       'Material_num', 'Size_num', 'Laptop Compartment_num', 'Waterproof_num',
       'Style_num', 'Color_num', 'Compartments/Weight', 'Weight/Compartments']

X = train[features]
X = X.fillna(X.mean())




#Selecting best features for modeling
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


xgb_model = XGBRegressor(n_estimators = 10, learning_rate = 0.5, max_depth = 3, eval_metric = 'mae')
xgb_model.fit(train_X, train_y)

xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
print('The accuracy of the model is: ', xgb_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', xgb_model.score(train_X, train_y))


xgb_final_model = XGBRegressor(n_estimators = 10, learning_rate = 0.5, max_depth = 3, eval_metric = 'mae')
xgb_final_model.fit(X,y)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

test['Brand_num'] = pd.factorize(test['Brand'])[0]
test['Material_num'] = pd.factorize(test['Material'])[0]
test['Size_num'] = pd.factorize(test['Size'])[0]
test['Laptop Compartment_num'] = pd.factorize(test['Laptop Compartment'])[0]
test['Waterproof_num'] = pd.factorize(test['Waterproof'])[0]
test['Style_num'] = pd.factorize(test['Style'])[0]
test['Color_num'] = pd.factorize(test['Color'])[0]

test['Compartments/Weight'] = test['Compartments']/test['Weight Capacity (kg)']
test['Weight/Compartments'] = test['Weight Capacity (kg)']/test['Compartments']



X_test = test[features]

y_pred = xgb_final_model.predict(X_test)


output = pd.DataFrame({'id': test['id'],
                       'prediction': y_pred})
output.to_csv('submission.csv', index=False)


output

