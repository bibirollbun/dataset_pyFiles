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
#Sklearn/RandomForest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# Load training data
train = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/train.csv')
train.info()


train.head()


train.describe()


#Cleaning/Reformating/Artificially Creating data
train['sex_num'] = pd.factorize(train['sex'])[0]
train['smoker_num'] = pd.factorize(train['smoker'])[0]
train['region_num'] = pd.factorize(train['region'])[0]


#Outlier removal
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

train = remove_outliers_iqr(train, 'children')
train = remove_outliers_iqr(train, 'bmi')

print('Done')


#Get list of all int/float values for modeling

train.select_dtypes(include=['int64','float64']).columns

#get list of all category values 
#train.select_dtypes(include=['object']).columns


#Create X,y values
y = train['charges']

all_features = ['age', 'bmi', 'children','sex_num', 'smoker_num',
       'region_num']

X = train[all_features]


#Generate a heatmap to see correlation betweeen variables

num_columns = ['age', 'bmi', 'children', 'charges', 'sex_num', 'smoker_num',
       'region_num'] #Put list of numberic variables here

correlation_matrix = train[num_columns].corr() #creates correlation matrix

sns.heatmap(correlation_matrix) #Generates heat map


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


feat_pvals['Feature'].values


#final features selected to use in modeling
features = ['age', 'bmi', 'children', 'smoker_num', 'region_num', 'sex_num']
X = train[features]

#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)
#print(X)


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Define base model
'''''xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define scorer (lower MAE is better, so set greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Use 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Run GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=mae_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(train_X, train_y)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best MAE (CV):", -grid_search.best_score_)'''''


#Add Parameters
model = XGBRegressor(colsample_bytree= 1.0, learning_rate= 0.1, max_depth= 5, n_estimators= 100, subsample= 0.8)
#model2 = RandomForestRegressor()

#Fitting data to model
model.fit(train_X, train_y)

#Making predictions on data
val_predictions = model.predict(val_X)

#Getting MAE/MSE/RMSE and Accuracy scores
val_mae = mean_absolute_error(val_predictions, val_y)
val_mse = mse = mean_squared_error(val_predictions, val_y)
val_rmse = np.sqrt(val_mse)

#Getting ROC_AUC_SCORE
#roc_auc = roc_auc_score(val_y, val_predictions)

#Printing results
print("Validation MAE for the model: {:,.0f}".format(val_mae))
print("Validation MSE for the model: {:,.0f}".format(val_mse))
print("Validation RMSE for the model: {:,.0f}".format(val_rmse))
print('The accuracy of the model is: ', model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', model.score(train_X, train_y))
#print('The AUC_ROC of the training model is: ', roc_auc)


#Create finalized model determined hyperparameters from above cell
final_model = XGBRegressor(colsample_bytree= 1.0, learning_rate= 0.1, max_depth= 5, n_estimators= 100, subsample= 0.8)
#Fit model to full training data
final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')
test['sex_num'] = pd.factorize(test['sex'])[0]
test['smoker_num'] = pd.factorize(test['smoker'])[0]
test['region_num'] = pd.factorize(test['region'])[0]
#Using the same features/variables as our train X value
X_test = test[features]
#X_test = scaler.fit_transform(X_test)
#Predicting test data's missing y target value
y_pred = final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'id': test['id'],
                       'charges': y_pred})
output.to_csv('submission.csv', index=False)


output

