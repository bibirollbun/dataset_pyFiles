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
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.info()


train.head()


#Factorize categorical variable:
train['Sex_num'] = pd.factorize(train['Sex'])[0]

#Create extra training data:
train['BMI'] = train['Weight']/train['Height']
train['BMI2'] = train['Weight'] / ((train['Height'] / 100) ** 2)
train['temp_to_hr_ratio'] = train['Body_Temp']/train['Heart_Rate']
train['Workout_Intensity'] = train['Heart_Rate'] * train['Duration']
train['Temp_duration'] = train['Body_Temp'] * train['Duration']



train.head()


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns


#Create X,y values
y = train['Calories']

features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Sex_num','Workout_Intensity',
       'BMI2','Temp_duration']

X = train[features]


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Add Parameters
model = XGBRegressor(n_estimators = 200, learning_rate = 0.1, max_depth = 10,eval_metric = 'mae')

#Fitting data to model
model.fit(train_X, train_y)

#Making predictions on data
val_predictions = model.predict(val_X)

#Getting MAE and Accuracy scores
val_mae = mean_absolute_error(val_predictions, val_y)

#Printing results
print("Validation MAE for Random Forest Model: {:,.0f}".format(val_mae))
print('The accuracy of the model is: ', model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model.score(train_X, train_y))


check = pd.DataFrame({
                       'Prediction': abs(val_predictions),
                      'Calories Actual': val_y})
check.head()


#Create finalized model determined hyperparameters from above cell
final_model = XGBRegressor(n_estimators = 200, learning_rate = 0.1, max_depth = 10,eval_metric = 'mae')
#Fit model to full training data
final_model.fit(X,y)




#load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test['Sex_num'] = pd.factorize(test['Sex'])[0]
test['BMI'] = test['Weight']/test['Height']
test['temp_to_hr_ratio'] = test['Body_Temp']/train['Heart_Rate']
test['BMI2'] = test['Weight'] / ((test['Height'] / 100) ** 2)
test['Workout_Intensity'] = test['Heart_Rate'] * test['Duration']
test['Temp_duration'] = test['Body_Temp'] * test['Duration']
#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'id': test['id'],
                       'Calories': abs(y_pred)})
output.to_csv('submission.csv', index=False)


output

