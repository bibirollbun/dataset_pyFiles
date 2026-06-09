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


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train.head()


train.info()


categorical_columns = train.select_dtypes(include='object').columns
print(categorical_columns)


train['Brand_numeric'] = pd.factorize(train['Brand'])[0]
train['Material_numeric'] = pd.factorize(train['Material'])[0]
train['Size_numeric'] = pd.factorize(train['Size'])[0]
train['Laptop Compartment_numeric'] = pd.factorize(train['Laptop Compartment'])[0]
train['Waterproof_numeric'] = pd.factorize(train['Waterproof'])[0]
train['Style_numeric'] = pd.factorize(train['Style'])[0]
train['Color_numeric'] = pd.factorize(train['Color'])[0]


train.info()


train.describe()


train = train.dropna()


train["CompWeightRatio"] = train['Compartments']/train['Weight Capacity (kg)']


train.head()


train.select_dtypes(include=['int64','float64']).columns


y = train.Price
features = ['Compartments', 'Weight Capacity (kg)', 'Brand_numeric',
       'Material_numeric', 'Size_numeric', 'Laptop Compartment_numeric',
       'Waterproof_numeric', 'Style_numeric', 'Color_numeric',
       'CompWeightRatio']
X = train[features]


#Looking for best features
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X,y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value')
feat_pvals[feat_pvals['p_value'] < 0.05]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Model 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print('The accuracy of the model is: ', rf_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', rf_model.score(train_X, train_y))


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df = df.dropna()
df = pd.get_dummies(data = df, drop_first = True)


df.head()


df.columns


y = df.Price
features = ['Compartments', 'Weight Capacity (kg)', 'Brand_Jansport',
       'Brand_Nike', 'Brand_Puma', 'Brand_Under Armour', 'Material_Leather',
       'Material_Nylon', 'Material_Polyester', 'Size_Medium', 'Size_Small',
       'Laptop Compartment_Yes', 'Waterproof_Yes', 'Style_Messenger',
       'Style_Tote', 'Color_Blue', 'Color_Gray', 'Color_Green', 'Color_Pink',
       'Color_Red']
X = df[features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Model 3
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=500)
xgb_model.fit(train_X, train_y,
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)],
             verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
print('The accuracy of the model is: ', xgb_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', xgb_model.score(train_X, train_y))


xgb_model_on_full_data = XGBRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
xgb_model_on_full_data.fit(X,y)

test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

test = pd.get_dummies(data = test, drop_first = True)

test_X = test[features].fillna(value = 0)

test_preds = xgb_model_on_full_data.predict(test_X)


output = pd.DataFrame({'id': test.id,
                       'Price': test_preds})
output.to_csv('submission.csv', index=False)


