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



train_file_path = '/kaggle/input/exploring-predictive-health-factors/train.csv'
train_data = pd.read_csv(train_file_path)
train_dummies = pd.get_dummies(data = train_data, drop_first = True)
train_dummies = train_dummies.dropna()




train_dummies.columns


#creating target variable
y = train_dummies['PCOS_Yes']
#adding all variables to features for now, will check for best fitting variables
features = ['ID', 'Weight_kg', 'Age_20-25', 'Age_25-30', 'Age_30-25', 'Age_30-35',
       'Age_30-40', 'Age_35-44', 'Age_45 and above', 'Age_Less than 20',
       'Age_Less than 20-25',
       'Hormonal_Imbalance_No, Yes, not diagnosed by a doctor',
       'Hormonal_Imbalance_Yes', 'Hormonal_Imbalance_Yes Significantly',
       'Hyperandrogenism_Yes', 'Hirsutism_No, Yes, not diagnosed by a doctor',
       'Hirsutism_Yes',
       'Conception_Difficulty_No, Yes, not diagnosed by a doctor',
       'Conception_Difficulty_Yes',
       'Conception_Difficulty_Yes, diagnosed by a doctor',
       'Insulin_Resistance_No, Yes, not diagnosed by a doctor',
       'Insulin_Resistance_Yes', 'Exercise_Frequency_3-4 Times a Week',
       'Exercise_Frequency_6-8 Times a Week', 'Exercise_Frequency_6-8 hours',
       'Exercise_Frequency_Less than 6 hours',
       'Exercise_Frequency_Less than usual', 'Exercise_Frequency_Never',
       'Exercise_Frequency_Rarely',
       'Exercise_Type_Cardio (e.g., running, cycling, swimming)',
       'Exercise_Type_Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)',
       'Exercise_Type_Cardio (e.g., running, cycling, swimming), None',
       'Exercise_Type_Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)',
       'Exercise_Type_Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)',
       'Exercise_Type_Flexibility and balance (e.g.',
       'Exercise_Type_Flexibility and balance (e.g., yoga, pilates)',
       'Exercise_Type_Flexibility and balance (e.g., yoga, pilates), None',
       'Exercise_Type_High-intensity interval training (HIIT)',
       'Exercise_Type_No Exercise', 'Exercise_Type_Somewhat',
       'Exercise_Type_Strength training',
       'Exercise_Type_Strength training (e.g.',
       'Exercise_Type_Strength training (e.g., weightlifting, resistance exercises)',
       'Exercise_Type_Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)',
       'Exercise_Duration_30 minutes',
       'Exercise_Duration_30 minutes to 1 hour',
       'Exercise_Duration_45 minutes',
       'Exercise_Duration_Less than 30 minutes',
       'Exercise_Duration_Less than 6 hours',
       'Exercise_Duration_More than 30 minutes',
       'Exercise_Duration_Not Applicable', 'Sleep_Hours_6-8 hours',
       'Sleep_Hours_9-12 hours', 'Sleep_Hours_Less than 6 hours',
       'Sleep_Hours_More than 12 hours', 'Exercise_Benefit_Not at All',
       'Exercise_Benefit_Somewhat', 'Exercise_Benefit_Yes Significantly']
X = train_dummies[features]




#Checking to see which variables are the best fit
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


#Settled on these variables after some trial and error 
features = ['Hyperandrogenism_Yes','Insulin_Resistance_Yes','Hormonal_Imbalance_Yes','Hirsutism_Yes']
X = train_dummies[features]


#splitting the training set
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)


#creating random forest regressor 
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
#checking accuracy scores
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print('The accuracy of the model is: ', rf_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', rf_model.score(train_X, train_y))



rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X,y)

#loading test data
test_data_path = '/kaggle/input/exploring-predictive-health-factors/test.csv'
test_data = pd.read_csv(test_data_path)
test_data = pd.get_dummies(data = test_data, drop_first = True)



#getting predictions 
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)



#creating submission
output = pd.DataFrame({'ID': test_data['ID'],
                       'PCOS': test_preds})
output.to_csv('submission.csv', index=False)

