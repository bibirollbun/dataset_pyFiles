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
#Sklearn/RandomForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler


# Load training data
train = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
train.info()


train.head()


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns


#Create X,y values
y = train['HOMELESS_RATE']

features = ['AGE_U18_PCT', 'AGE_18_24_PCT', 'AGE_25_34_PCT',
       'AGE_35_44_PCT', 'AGE_45_54_PCT', 'AGE_55_59_PCT', 'AGE_60_61_PCT',
       'AGE_62_64_PCT', 'AGE_65_69_PCT', 'AGE_70_79_PCT', 'AGE_80_PLUS_PCT',
       'AGE_25_PLUS_PCT', 'FAMILY_MEMBERS_UNDER_18_PCT', 'RACE_WHITE_NH_PCT',
       'RACE_BLACK_NH_PCT', 'RACE_NATIVE_NH_PCT', 'RACE_ASIAN_NH_PCT',
       'RACE_PACIFIC_NH_PCT', 'RACE_TWO_OR_MORE_NH_PCT',
       'RACE_HISPANIC_ANY_PCT', 'VETERAN_POP_PCT', 'NONVETERAN_POP_PCT',
       'DISABILITY_POP_PCT', 'NODISABILITY_POP_PCT', 'TOTAL_HOUSEHOLDS_PCT',
       'FAMILY_HH_TOTAL', 'FAMILY_HH_CHILD_LT18_PCT',
       'NONFAMILY_SINGLE_MALE_PCT', 'NONFAMILY_SINGLE_FEMALE_PCT',
       'MULTI_PERSON_NONFAMILY_HH_PCT', 'INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']

X = train[features]


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


features = ['AGE_18_24_PCT','RACE_BLACK_NH_PCT']
X = train[features]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Add Parameters
model = XGBRegressor()
#model2 = RandomForestRegressor()

#Fitting data to model
model.fit(train_X, train_y)

#Making predictions on data
val_predictions = model.predict(val_X)

#Getting MAE and Accuracy scores
val_mae = mean_absolute_error(val_predictions, val_y)

#Printing results
print("Validation MAE for the model: {:,.0f}".format(val_mae))
print('The accuracy of the model is: ', model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model.score(train_X, train_y))


#Create finalized model determined hyperparameters from above cell
final_model = XGBRegressor()
#Fit model to full training data
final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'ID': test['ID'],
                       'HOMELESS_RATE': y_pred})
output.to_csv('submission.csv', index=False)

