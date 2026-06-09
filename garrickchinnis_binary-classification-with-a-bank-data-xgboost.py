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
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score


# Load training data
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
train.info()


train.describe()


train.head()


sns.histplot(data=train, x = 'age')


bins = [0, 12, 19, 65, np.inf] # Defines ranges: 0-12 (exclusive upper), 13-19, 20-65, 66+
labels = ["Child", "Teenager", "Adult", "Elderly"]
train['age_category'] = pd.cut(x=train['age'], bins=bins, labels=labels, right=True)


bins2 = [-10000,0,1000,10000,50000]
labels2 = ["Negative", "SomeMoney", "WellOff", "Rich"]
train['balance_category'] = pd.cut(x=train['balance'], bins=bins2, labels=labels2, right=True)


train.select_dtypes(include=['object','category']).columns


train['job'] = pd.factorize(train['job'])[0]
train['marital'] = pd.factorize(train['marital'])[0]
train['education'] = pd.factorize(train['education'])[0]
train['default'] = pd.factorize(train['default'])[0]
train['housing'] = pd.factorize(train['housing'])[0]
train['loan'] = pd.factorize(train['loan'])[0]
train['contact'] = pd.factorize(train['contact'])[0]
train['month'] = pd.factorize(train['month'])[0]
train['poutcome'] = pd.factorize(train['poutcome'])[0]
train['age_category'] = pd.factorize(train['age_category'])[0]
train['balance_category'] = pd.factorize(train['balance_category'])[0]


train.select_dtypes(include=['int64','float64']).columns


import seaborn as sns
num_columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
       'pdays', 'previous', 'poutcome', 'y','age_category','balance_category']
correlation_matrix = train[num_columns].corr()
sns.heatmap(correlation_matrix)


#Create X,y values
y = train['y']

features = ['balance','duration',
       'pdays', 'previous', 'poutcome','age_category','month']


X = train[features]


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


feat_pvals['Feature'].values


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Add Parameters
model = XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 9, eval_metric = 'auc')
#model2 = RandomForestRegressor()

#Fitting data to model
model.fit(train_X, train_y)

#Making predictions on data
val_predictions = model.predict(val_X)

#Getting MAE and Accuracy scores
val_mae = mean_absolute_error(val_predictions, val_y)

#Getting ROC_AUC_SCORE
roc_auc = roc_auc_score(val_y, val_predictions)

#Printing results
#print("Validation MAE for the model: {:,.0f}".format(val_mae))
#print('The accuracy of the model is: ', model.score(val_X, val_y)) 
#print('The accuracy of the training model is: ', model.score(train_X, train_y))
print('The AUC_ROC of the training model is: ', roc_auc)


'''model2 = RandomForestClassifier()
model2.fit(train_X, train_y)
val_predictions = model2.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for the model: {:,.0f}".format(val_mae))
print('The accuracy of the model is: ', model2.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model2.score(train_X, train_y))'''


'''model3 = XGBClassifier()
model3.fit(train_X, train_y)
val_predictions = abs(model3.predict(val_X))
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for the model: {:,.0f}".format(val_mae))
print('The accuracy of the model is: ', model3.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model3.score(train_X, train_y))'''


#Create finalized model determined hyperparameters from above cell
final_model = model
#Fit model to full training data
final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test['age_category'] = pd.cut(x=test['age'], bins=bins, labels=labels, right=True)
test['balance_category'] = pd.cut(x=test['balance'], bins=bins2, labels=labels2, right=True)
test['job'] = pd.factorize(test['job'])[0]
test['marital'] = pd.factorize(test['marital'])[0]
test['education'] = pd.factorize(test['education'])[0]
test['default'] = pd.factorize(test['default'])[0]
test['housing'] = pd.factorize(test['housing'])[0]
test['loan'] = pd.factorize(test['loan'])[0]
test['contact'] = pd.factorize(test['contact'])[0]
test['month'] = pd.factorize(test['month'])[0]
test['poutcome'] = pd.factorize(test['poutcome'])[0]
test['age_category'] = pd.factorize(test['age_category'])[0]
test['balance_category'] = pd.factorize(test['balance_category'])[0]
#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'id': test['id'],
                       'prediction': y_pred})
output.to_csv('submission.csv', index=False)


output

