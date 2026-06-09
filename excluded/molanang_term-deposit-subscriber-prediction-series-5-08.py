!pip install --upgrade xgboost


!pip install scikit-learn==1.5.0


import pandas as pd
import numpy as np
from datetime import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

print('---------- Train info: ----------')
print(df_train.info())
print('\n---------- Test info: ----------')
print(df_test.info())


### Rename Several Columns to Improve Readability ###

df_train.rename(columns={'balance': 'avg_yearly_balance_eur', 
                         'housing': 'housing_loan',
                         'loan': 'personal_loan',
                         'contact': 'communication_contact',
                         'day': 'last_contact_day_of_month',
                         'month': 'last_contact_month',
                         'duration': 'last_contact_duration_second',
                         'campaing': '#_contact_this_campaign',
                         'pdays': '#_days_since_last_contact',
                         'previous': '#_contact_before_this_campaign',
                         'poutcome': 'previous_campaign_outcome',
                         'y': 'subscribed_term_deposit'}, 
                inplace=True)
df_test.rename(columns={'balance': 'avg_yearly_balance_eur', 
                         'housing': 'housing_loan',
                         'loan': 'personal_loan',
                         'contact': 'communication_contact',
                         'day': 'last_contact_day_of_month',
                         'month': 'last_contact_month',
                         'duration': 'last_contact_duration_second',
                         'campaing': '#_contact_this_campaign',
                         'pdays': '#_days_since_last_contact',
                         'previous': '#_contact_before_this_campaign',
                         'poutcome': 'previous_campaign_outcome',
                         'y': 'subscribed_term_deposit'}, 
                inplace=True)


### Quick EDA of the Data ###

'''
ProfileReport from ydata_profiling is used because it is simple and efficient to use. 
It can generates some fundamentals EDA steps, such as the distribution of each variable, 
correlation between each numerical variable in the data, etc without having to type a lot of code. 
It is very helpful to use.
'''

EDA_report = ProfileReport(df_train, title='Bank Marketing Data EDA') # using train data
EDA_report


### Multicollinearity Checking ###

# Create a dataframe that contains only the numerical features
df_num = df_train.select_dtypes('number')
df_num.drop(columns='subscribed_term_deposit', inplace=True)

# Create a dataframe that contains the VIF values for each numerical feature
df_vif = pd.DataFrame()
df_vif['Features'] = df_num.columns
df_vif['VIF'] = [vif(df_num, i) for i in range(df_num.shape[1])]

display(df_vif)


### Split the Target Variable from the Features in Train Data ###

y_train = df_train[['subscribed_term_deposit']]
df_train_feat = df_train.drop(columns='subscribed_term_deposit')

print(f'The shape of train feature data: {df_train_feat.shape}')
print(f'The shape of train target variable: {y_train.shape}')


### Split between the Numerical Features and Categorical Features ###

df_train_num = df_train_feat.select_dtypes('number').drop(columns='id')
df_train_cat = df_train.select_dtypes('object')
df_test_num = df_test.select_dtypes('number').drop(columns='id')
df_test_cat = df_test.select_dtypes('object')

print(f'The shape of train numerical feature data: {df_train_num.shape}')
print(f'The shape of train categorical feature data: {df_train_cat.shape}')
print(f'The shape of test numerical feature data: {df_test_num.shape}')
print(f'The shape of test categorical feature data: {df_test_cat.shape}')


### Scale the Numerical Features from Both Train and Test Data ###

scaler = RobustScaler()

df_train_num_scaled = scaler.fit_transform(df_train_num)
df_test_num_scaled = scaler.transform(df_test_num)

X_train_scaled_num = pd.DataFrame(df_train_num_scaled, 
                                  columns = df_train_num.columns,
                                  index = df_train_num.index)
X_test_scaled_num = pd.DataFrame(df_test_num_scaled, 
                                 columns = df_test_num.columns,
                                 index = df_test_num.index)

print(f'Shape of train numerical feature data after scaling: \n{X_train_scaled_num.shape}')
print('Train data after scaling preview: ')
display(X_train_scaled_num.head())
print(f'\n\nShape of test numerical feature data after scaling: \n{X_test_scaled_num.shape}')
print('Test data after scaling preview: ')
display(X_test_scaled_num.head())


### Create Dummies for the Categorical Features ###

X_train_dummy = pd.get_dummies(df_train_cat)
X_test_dummy = pd.get_dummies(df_test_cat)

print(f'Shape of train categorical feature data: \n{X_train_dummy.shape}')
print('Train data after dummification preview: ')
display(X_train_dummy.head())
print(f'\n\nShape of test categorical feature data: \n{X_test_dummy.shape}')
print('Test data after dummification preview: ')
display(X_test_dummy.head())


### Combine both Numerical and Categorical Features into One Data Frame ###

X_train = pd.concat([X_train_scaled_num, X_train_dummy], axis=1)
X_test = pd.concat([X_test_scaled_num, X_test_dummy], axis=1)

print(f'Shape of train data after recombination: {X_train.shape}')
print(f'Shape of test data after recombination: {X_test.shape}')


### Model Building & Performance in Training Set ###

warnings.filterwarnings('ignore')

lgb_mod = LGBMClassifier(random_state=123, verbose=-1)

# Measure training model running time
start_time = time.time()
lgb_mod.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time

# Measure model prediction time
start_time = time.time()
y_train_lgb_pred = lgb_mod.predict(X_train)
end_time = time.time()
prediction_time = end_time - start_time

# Compute ROC AUC score
lgb_mod_train_auc = roc_auc_score(y_train, y_train_lgb_pred)

print('Light GBM model training set performances: ')
print(f'AUC score: {lgb_mod_train_auc}')

print(f'\nModel training time: {train_time:.3f} seconds')
print(f'Model prediction time: {prediction_time:.3f} seconds')


### Validation Set Performance ###

lgb_mod_val_auc = cross_val_score(lgb_mod, X_train, y_train,
                                  scoring="roc_auc", cv=5)

print('Light GBM model validation set performance: ')
print(f'Average AUC score: {np.mean(lgb_mod_val_auc)}')


### Hyperparameters Tuning ###

warnings.filterwarnings('ignore')

lgb_params = {
         'num_leaves': [50, 150, 300],  
         'learning_rate': [0.01, 0.05, 0.3],
         'n_estimators': [100, 200, 300],  
         'max_depth': [3, 5, 12],  
         'feature_fraction': [0.5, 0.8, 1.0],
         'bagging_fraction': [0.5, 0.8, 1.0],
         'lambda_l1': [0, 0.1, 5.0],
         'lambda_l2': [0, 0.1, 5.0]
     }

# Halving grid search CV is used instead of grid search CV because it is faster in terms of computing time
lgb_halving_grid_search = HalvingGridSearchCV(
     estimator=lgb_mod,
     param_grid=lgb_params,
     cv=5,
     scoring='roc_auc',
     random_state=333
)

# Measure running time of the model
start_time = time.time()
lgb_halving_grid_search.fit(X_train, y_train)
end_time = time.time()
running_time = end_time - start_time

# print halving grid search cv results
print("Light GBM halving grid search CV best parameters:", lgb_halving_grid_search.best_params_)
print("Best mean score:", lgb_halving_grid_search.best_score_)

# print running time result
print(f'\nModel running time: {running_time/60:.2f} minutes')


### Model Building & Performance in Training Set ###

warnings.filterwarnings('ignore')

xgb_mod = XGBClassifier(objective='binary:logistic', 
                               random_state=123)

# Measure training model running time
start_time = time.time()
xgb_mod.fit(X_train, y_train)
end_time = time.time()
train_time = end_time - start_time

# Measure model prediction time
start_time = time.time()
y_train_xgb_pred = xgb_mod.predict(X_train)
end_time = time.time()
prediction_time = end_time - start_time

# Compute ROC AUC score
xgb_mod_train_auc = roc_auc_score(y_train, y_train_xgb_pred)

print('XGB model training set performances: ')
print(f'AUC score: {xgb_mod_train_auc}')

print(f'\nModel training time: {train_time:.3f} seconds')
print(f'Model prediction time: {prediction_time:.3f} seconds')


### Validation Set Performance ###

xgb_mod_val_auc = cross_val_score(xgb_mod, X_train, y_train,
                                  scoring='roc_auc', cv=5, error_score='raise')

print('XGB model validation set performance: ')
print(f'Average AUC score: {np.mean(xgb_mod_val_auc)}')


### Hyperparameters Tuning ###

warnings.filterwarnings('ignore')

xgb_params = {
         'n_estimators': [150, 500, 750, 900],
         'learning_rate': [0.01, 0.1, 0.5, 0.8],
         'max_depth': [3, 5, 7],
         'subsample': [0.8, 1.0],
         'colsample_bytree': [0.8, 1.0]
     }

# Halving grid search CV is used instead of grid search CV because it is faster in terms of computing time
xgb_halving_grid_search = HalvingGridSearchCV(
     estimator=xgb_mod,
     param_grid=xgb_params,
     cv=5,
     scoring='roc_auc',
     random_state=333
)

# Measure running time of the model
start_time = time.time()
xgb_halving_grid_search.fit(X_train, y_train)
end_time = time.time()
running_time = end_time - start_time

# print halving grid search cv results
print("XGB halving grid search CV best parameters:", xgb_halving_grid_search.best_params_)
print("Best mean score:", xgb_halving_grid_search.best_score_)

# print running time result
print(f'\nModel running time: {running_time/60:.2f} minutes')


### Light GBM Model Prediction on Train Data ###

# Predict the train data using the features
lgb_y_pred = lgb_halving_grid_search.predict(X_train)

# Create a classification report for the model
lgb_report = classification_report(y_train, lgb_y_pred)

print(f'LGB model report: \n{lgb_report}')


### XGB Model Prediction on Train Data ###

# Predict the train data using the features
xgb_y_pred = xgb_halving_grid_search.predict(X_train)

# Create a classification report for the model
xgb_report = classification_report(y_train, xgb_y_pred)

print(f'XGB model report: \n{xgb_report}')


### Predicting the Probability of the Test Data ###

# Predict the test data
xgb_test_pred_proba = xgb_halving_grid_search.predict_proba(X_test)

# Create a data frame for the prediction results
xgb_test_df = pd.DataFrame(xgb_test_pred_proba[:, 1],
                           index = X_test.index,
                           columns = ['y']).\
              reset_index().\
              rename(columns={'index': 'id'})

xgb_test_df['id'] = df_test['id']

# Export as .csv file
xgb_test_df.to_csv("submission.csv", index=False)

display(xgb_test_df.head(10))

