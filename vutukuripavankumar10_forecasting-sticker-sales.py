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


df = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv',index_col='id',parse_dates=['date'])


df.head()


df.sample(10)


df.tail()


df.info() 


df.shape


#finding the null values
df.isnull().sum()   # num_sold has a missing values


df.isnull().mean()*100


df.describe()


#finding the duplicates, 
df.duplicated().sum() #no duplicates found


#count of stores
df['store'].value_counts()


df['country'].value_counts()


df['country'].unique() #the unique countries


#count of products
df['product'].value_counts()


#value counts of date column
df['date'].value_counts()


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


sns.countplot(data=df,x='country')


sns.countplot(data=df,x='store');


sns.countplot(data=df,x='product')


sns.relplot(data=df,y='num_sold',x='date'); #scatterplot of trends


sns.histplot(data=df,x='num_sold',bins=50)


sns.displot(data=df,x='num_sold',kind='kde',fill=True) #distribution plot of num_sold


!pip install ydata-profiling


from ydata_profiling import ProfileReport 
prof = ProfileReport(df) 
prof.to_file(output_file='output_file.html')


prof.to_widgets()


#removing the missing rows using cca(complete case analysis)
df.dropna(inplace=True)


df.shape


#working on date column
df['date_year'] = df['date'].dt.year #extract year
df['date_month_no'] = df['date'].dt.month #extract month number
df['date_dow_no'] = df['date'].dt.dayofweek #extract dayofweek number


df.info()


df.head()


#changing the column-num_sold to 7 index
col_to_move = 'num_sold'
new_position = 7  # Index where you want to place the column

col = df.pop(col_to_move)  # Remove the column
df.insert(new_position, col_to_move, col)  # Insert at new position



df.shape


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler


X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:7],df.iloc[:,-1],test_size=0.2,random_state=2)


X_train.shape


y_train.shape
y_train.sample(5)


X_train.head()


#applying OHE to country,store,and product
transformer = ColumnTransformer([
    ('trnf1',OneHotEncoder(sparse_output=False,drop='first'),['country','store','product']),
],remainder='passthrough'
)


X_train1 = transformer.fit_transform(X_train) #transformed X_train


X_train.shape
X_train.head()


X_test1 = transformer.transform(X_test) #transformed X_test


X_test.shape


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train1,y_train)


y_pred = lr.predict(X_test1)


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


print('R2_score',r2_score(y_test,y_pred))
print('mean_absolute_error',mean_absolute_error(y_test,y_pred))
print('mean_squared_error',mean_squared_error(y_test,y_pred))


#using randomforestregressor
'''
from sklearn.ensemble import RandomForestRegressor 
rfr = RandomForestRegressor(n_estimators=200,random_state=2,oob_score=True)
rfr.fit(X_train1,y_train)
rf_predict = rfr.predict(X_test1)
'''


'''
print('R2_score',r2_score(y_test,rf_predict))
print('mean_absolute_error',mean_absolute_error(y_test,rf_predict))
print('mean_squared_error',mean_squared_error(y_test,rf_predict))`

rfr.oob_score_

'''


''''
#applying gridsearchcv
from sklearn.model_selection import GridSearchCV
param_dict = {
    'n_estimators':[100,200],
    'criterion':['squared_error','absolute_error']
    
}

rfr = RandomForestRegressor()

grid = GridSearchCV(rfr,param_grid=param_dict,cv=3,n_jobs=-1)

grid.fit(X_train1,y_train)

grid.best_estimator_

grid.best_params_

grid.best_score_
'''


#applying randomsearchcv
'''
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'n_estimators':[100,200],
    'max_features':[0.2,0.4,1],
    'max_depth':[2,8,None],
    'max_samples':[0.5,0.75,1],
    'bootstrap':[True,False],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]
}
'''


'''
rfr_grid = RandomizedSearchCV(estimator=rfr,param_distributions=param_grid,cv=5,verbose=2,n_jobs=-1)

rfr_grid.fit(X_train1,y_train)

rfr_grid.best_params_

rfr_grid.best_score_
'''


!pip install cmaes


!pip install optuna


!pip install xgboost


#applying optuna
import optuna
from optuna.samplers import CmaEsSampler 
import xgboost as xgb


#creating the sampler
sampler = CmaEsSampler()


'''
#create objective function
def objective(trial):

    #choose the algorithm to tune
    classifier_name = trial.suggest_categorical('classifier',['RandomForest','GradientBoosting','XgBoost'])
    
    if classifier_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators',50,100)
        max_depth = trial.suggest_int('max_depth',3,5)
        min_samples_split = trial.suggest_int('min_samples_split',2,5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',1,5)
        bootstrap=trial.suggest_categorical('bootstrap',[True,False])
        random_state=2

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=2
        )

    elif classifier_name == 'GradientBoosting':
        #gradientboosting hyperparameters
        n_estimators = trial.suggest_int('n_estimators',50,100)
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3)

        model= GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

    elif classifier_name == 'XgBoost':

        #XgBoost hyperparameters 
        n_estimators = trial.suggest_int('n_estimators',50,500)
        learning_rate = trial.suggest_float('learning_rate',0.01,0.3)
        max_depth = trial.suggest_int('max_depth',3,7)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=2
        )
    
    #perform cross-validation and return the mean accuracy
    score = cross_val_score(model,X_train1,y_train,cv=3,scoring='r2',n_jobs=-1).mean()
    return score


'''
#create a study and optimize it using CmaEsSampler
study = optuna.create_study(direction='maximize',sampler=sampler)
study.optimize(objective,n_trials=20,n_jobs=-1)


'''
#Retrieve the best trial
best_trial = study.best_trial 
print('Best trial parameters:',best_trial.params)
print('Best trial r2_score:',best_trial.value)


'''
study.trials_dataframe()



#study.trials_dataframe()['params_classifier'].value_counts()


#study.trials_dataframe().groupby('params_classifier')['value'].mean()


xgb_model = xgb.XGBRegressor(objective='reg:absoluteerror',n_estimators=1000,learning_rate= 0.10052016177571975, max_depth= 7)


xgb_model.fit(X_train1,y_train)


y_pred_xg = xgb_model.predict(X_test1) 



mae = mean_absolute_error(y_test,y_pred_xg)
print(mae)
print('r2-score:',r2_score(y_test,y_pred_xg))


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


test.head()


test['date'] = pd.to_datetime(test['date'])


#feature engineering for the data
test.dropna(inplace=True)


#working on date column
test['date_year'] = test['date'].dt.year #extract year
test['date_month_no'] = test['date'].dt.month #extract month number
test['date_dow_no'] = test['date'].dt.dayofweek #extract dayofweek number


test.info()


test.head()


test.drop('id',axis=1,inplace=True)


test.drop('date',axis=1,inplace=True)


#applying OHE to country,store,and product
transformer_test = ColumnTransformer([
    ('trnf1',OneHotEncoder(sparse_output=False,drop='first'),['country','store','product']),
],remainder='passthrough'
)


test.shape


test.head()


test_a = transformer.transform(test)


pred_sub = xgb_model.predict(test_a)


print(pred_sub)


test_sub = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
test_sub['num_sold'] = pred_sub
test_sub[['id','num_sold']].to_csv('submission_1.csv',index=False)




