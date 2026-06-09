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


train_comb = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
train_func = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
train_quan = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')
train_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')


train_comb.head()
#Target variables


train_func.head()
#numerical data


train_func.select_dtypes(include=['object']).columns


train_quan.head()


train_cat.head()


train = pd.merge(train_comb, train_quan, on='participant_id', how='inner')
train = pd.merge(train, train_cat, on='participant_id', how='inner')


#Get list of all int/float values for model
train.select_dtypes(include=['int64','float64']).columns


train = train.dropna()


#Create X,y values
y = train['Sex_F']

features = ['EHQ_EHQ_Total', 'ColorVision_CV_Score',
       'APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV',
       'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP',
       'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total',
       'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing',
      'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity',
       'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial',
       'MRI_Track_Age_at_Scan', 'Basic_Demos_Enroll_Year',
       'Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity',
       'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location',
       'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ',
       'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']

X = train[features]


X.info()


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


feat_pvals['Feature'].values


features = ['SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Emotional_Problems',
       'ColorVision_CV_Score', 'SDQ_SDQ_Externalizing',
       'Barratt_Barratt_P1_Edu', 'SDQ_SDQ_Prosocial',
       'SDQ_SDQ_Internalizing']
X = train[features]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Create initial training model
#Can hypertune model with the above parameters
xgb_model = XGBRegressor(n_estimators = 10, max_depth = 3)

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
xgb_final_model = XGBRegressor()
#Fit model to full training data
xgb_final_model.fit(X,y)


#load test data
test_cat = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
test_quan = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')
test = pd.merge(test_cat, test_quan, on='participant_id', how='inner')

#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = xgb_final_model.predict(X_test)


test['Sex_F'] = y_pred


y2 = train['ADHD_Outcome']
features2 = ['Sex_F', 'EHQ_EHQ_Total', 'ColorVision_CV_Score',
       'APQ_P_APQ_P_CP', 'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV',
       'APQ_P_APQ_P_OPD', 'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP',
       'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Difficulties_Total',
       'SDQ_SDQ_Emotional_Problems', 'SDQ_SDQ_Externalizing',
       'SDQ_SDQ_Generating_Impact', 'SDQ_SDQ_Hyperactivity',
       'SDQ_SDQ_Internalizing', 'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial',
       'MRI_Track_Age_at_Scan', 'Basic_Demos_Enroll_Year',
       'Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity',
       'PreInt_Demos_Fam_Child_Race', 'MRI_Track_Scan_Location',
       'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ',
       'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ']
X2 = train[features2]


train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state=1)


#Fitting data to model
xgb_model.fit(train_X2, train_y2)

#Making predictions on data
xgb_val_predictions2 = xgb_model.predict(val_X2)

#Getting MAE and Accuracy scores
xgb_val_mae = mean_absolute_error(xgb_val_predictions2, val_y2)

#Printing results
print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
print('The accuracy of the model is: ', xgb_model.score(val_X2, val_y2)) 
print('The accuracy of the training model is: ', xgb_model.score(train_X2, train_y2))


xgb_final_model2 = XGBRegressor(n_estimators = 50, max_depth = 3)
#Fit model to full training data
xgb_final_model2.fit(X2,y2)

X_test2 = test[features2]

#Predicting test data's missing y target value
y_pred2 = xgb_final_model2.predict(X_test2)


#Generic submission formatting 

output = pd.DataFrame({'participant_id': test['participant_id'], 'ADHD_Outcome': y_pred2.round(),
                       'Sex_F': y_pred.round()})
output.to_csv('submission.csv', index=False)


output

