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
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train.info()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test.info()


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns


#Create X,y values
y = train['rainfall']

features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
       'windspeed']

X = train[features]


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


feat_pvals['Feature'].values


features = ['cloud', 'sunshine', 'humidity', 'windspeed', 'dewpoint',
       'maxtemp', 'pressure', 'temparature']
X = train[features]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Create initial training model
#Can hypertune model with the above parameters
xgb_model = XGBRegressor(learning_rate = 0.1, max_depth = 3,eval_metric = 'mae')

#Fitting data to model
xgb_model.fit(train_X, train_y)

#Making predictions on data
xgb_val_predictions = xgb_model.predict(val_X)

#Getting MAE and Accuracy scores
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

#Printing results
print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
print('The accuracy of the model is: ', xgb_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', xgb_model.score(train_X, train_y))


#Create finalized model determined hyperparameters from above cell
xgb_final_model = XGBRegressor(learning_rate = 0.1, max_depth = 3,eval_metric = 'mae')
#Fit model to full training data
xgb_final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = xgb_final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'id': test['id'],
                       'prediction': y_pred})
output.to_csv('submission.csv', index=False)


output

