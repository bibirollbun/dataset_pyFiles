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
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# Load training data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train.info()


encoder = LabelEncoder()
train['Personality_Encoded'] = encoder.fit_transform(train['Personality'])
train['Stage_fear_n'] = pd.factorize(train['Stage_fear'])[0]
train['Drained_after_socializing_n'] = pd.factorize(train['Drained_after_socializing'])[0]
#train = pd.get_dummies(train, columns=['Stage_fear','Drained_after_socializing'])



train.head()


original_num_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'Personality_Encoded',
       'Stage_fear_n', 'Drained_after_socializing_n']


original_correlation_matrix = train[original_num_columns].corr()
sns.heatmap(original_correlation_matrix)


#Extra Data
train['Score'] = train['Time_spent_Alone'] + train['Social_event_attendance'] + train['Going_outside'] + train['Friends_circle_size'] + train['Post_frequency']
train['Alone_Stage_fear_Score'] = train['Time_spent_Alone'] + train['Stage_fear_n']
train['Drained_after_socializing_alone'] = train['Time_spent_Alone'] + train['Drained_after_socializing_n']
train['Lonely'] = (train['Time_spent_Alone'] < 3).astype(int)
train['Social'] = (train['Social_event_attendance'] > 5).astype(int)
train['Popular'] = (train['Friends_circle_size'] > 7).astype(int)
train['Outdoorsie'] = (train['Going_outside'] > 4).astype(int)
train['Social_Media_Proficient'] = (train['Post_frequency'] > 4).astype(int)


#'Time_spent_Alone','Stage_fear_n','Drained_after_socializing_n'


train.describe()


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns


numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency', 'Personality_Encoded',
       'Stage_fear_n', 'Drained_after_socializing_n', 'Score','Alone_Stage_fear_Score', 'Drained_after_socializing_alone','Lonely',
       'Social', 'Popular', 'Outdoorsie', 'Social_Media_Proficient']
correlation_matrix = train[numeric_columns].corr()
sns.heatmap(correlation_matrix)


#Create X,y values
y = train['Personality_Encoded']

#features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       #'Friends_circle_size', 'Post_frequency',
       #'Stage_fear_n', 'Drained_after_socializing_n', 'Score','Alone_Stage_fear_Score', 'Drained_after_socializing_alone','Lonely',
       #'Social', 'Popular', 'Outdoorsie', 'Social_Media_Proficient']
features = ['Stage_fear_n', 'Drained_after_socializing_n', 'Time_spent_Alone','Alone_Stage_fear_Score', 'Drained_after_socializing_alone']

X = train[features]
mean = X.mean()


X = X.fillna(mean)


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Add Parameters
model = XGBClassifier(objective = 'binary:logistic',n_estimators = 80, learning_rate = 0.2,  max_depth = 10,eval_metric = 'auc',tree_method = 'auto')
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


#model2 = RandomForestClassifier(n_estimators = 100,max_depth = 3)

#Fitting data to model
#model2.fit(train_X, train_y)

#Making predictions on data
#val_predictions2 = model2.predict(val_X)

#Getting MAE and Accuracy scores
#val_mae2 = mean_absolute_error(val_predictions, val_y)

#Printing results
#print("Validation MAE for the model: {:,.0f}".format(val_mae2))
#print('The accuracy of the model is: ', model2.score(val_X, val_y)) 
#print('The accuracy of the training model is: ', model2.score(train_X, train_y))


#Create finalized model determined hyperparameters from above cell
final_model = XGBClassifier(objective = 'binary:logistic',n_estimators = 80, learning_rate = 0.2,  max_depth = 10,eval_metric = 'auc',tree_method = 'auto')
#Fit model to full training data
final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test['Stage_fear_n'] = pd.factorize(test['Stage_fear'])[0]
test['Drained_after_socializing_n'] = pd.factorize(test['Drained_after_socializing'])[0]
test['Alone_Stage_fear_Score'] = test['Time_spent_Alone'] + test['Stage_fear_n']
test['Drained_after_socializing_alone'] = test['Time_spent_Alone'] + test['Drained_after_socializing_n']
test['Score'] = test['Time_spent_Alone'] + test['Social_event_attendance'] + test['Going_outside'] + test['Friends_circle_size'] + test['Post_frequency']
test['Lonely'] = (test['Time_spent_Alone'] < 3).astype(int)
test['Social'] = (test['Social_event_attendance'] > 5).astype(int)
test['Popular'] = (test['Friends_circle_size'] > 7).astype(int)
test['Outdoorsie'] = (test['Going_outside'] > 4).astype(int)
test['Social_Media_Proficient'] = (test['Post_frequency'] > 4).astype(int)
#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)

test['Personality'] = encoder.inverse_transform(y_pred)


#Generic submission formatting 

output = pd.DataFrame({'id': test['id'],
                       'Personality': test['Personality']})
output.to_csv('submission.csv', index=False)


output

