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


# Import Packages
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection  import train_test_split,KFold
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from bayes_opt import BayesianOptimization

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv',index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv',index_col='id')


# concat data 
train_data['label'] = 'train'
test_data['label'] = 'test'
concat_data = pd.concat([train_data,test_data])


#  split cls_cols and num_cols  
target = 'Listening_Time_minutes'
cls_cols = concat_data.select_dtypes(include=['object']).drop(columns=['label']).columns
num_cols = concat_data.select_dtypes(include=['float']).drop(columns=target).columns


# check columns if including loss value 
concat_data.info()
# We can find that there are there num_cols including loss value(1.Episode_Length_minutes,2.Guest_Popularity_percentage,3.Number_of_Ads)


#  Assign values to extremely large outliers
for i in num_cols:
    Q1 = concat_data[i].quantile(0.25)
    Q3 = concat_data[i].quantile(0.75)
    IQR = Q3 - Q1
    MAX_IQR = Q3 + 1.5 * IQR
    concat_data.loc[concat_data[i] >= MAX_IQR,i] = MAX_IQR


# Assign values to the missing values using the median.
loss_cols = ['Episode_Length_minutes','Guest_Popularity_percentage','Number_of_Ads']
for i in loss_cols:
    concat_data[i] = concat_data[i].fillna(concat_data[i].median())


# Add New Features 
# 1.avg_Popularity_percentage
concat_data['avg_Popularity_percentage'] = (concat_data['Host_Popularity_percentage'] + concat_data['Guest_Popularity_percentage']) / 2
# 2. avg_Ads_time
concat_data['avg_Ads_time'] = concat_data['Episode_Length_minutes'] / np.where(concat_data['Number_of_Ads']==0,1,concat_data['Number_of_Ads'])
# 3. concat 'Publication_Day' and 'Publication_Time'
concat_data['Day_Time'] = concat_data['Publication_Day'] + '_' + concat_data['Publication_Time']
# 5. mult_Popularity_percentage
concat_data['mult_Popularity_percentage'] = concat_data['Host_Popularity_percentage'] * concat_data['Guest_Popularity_percentage']


# Encoding cls_nums
target = 'Listening_Time_minutes'
cls_cols = concat_data.select_dtypes(include=['object']).drop(columns=['label']).columns
num_cols = concat_data.select_dtypes(include=['float']).drop(columns=target).columns

encoder = OrdinalEncoder()
concat_data[cls_cols] = encoder.fit_transform(concat_data[cls_cols])


# Data standardization
scaler = StandardScaler()
concat_data[num_cols] = scaler.fit_transform(concat_data[num_cols])


# split data to train and test 
new_train_data = concat_data[concat_data['label']=='train'].drop(columns='label',axis=1)
new_test_data = concat_data[concat_data['label']=='test'].drop(columns=['label','Listening_Time_minutes'],axis=1)


# split train data 
X = new_train_data.drop(columns=target,axis=1)
y = new_train_data[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=42)


# Using Bayes 
def xgb_evaluate(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eval_metric': 'rmse'
    }

    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
    
    # Obtain the best validation score
    best_score = model.best_score
    return -best_score  

# Define the parameter range
pbounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'gamma': (0, 5),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)


# model training 
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
print (best_params)

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# 绘制特征重要性
xgb.plot_importance(final_model)


# model predict 
y_pred_xgb = final_model.predict(new_test_data)


# output predict results 
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = y_pred_xgb
submission.to_csv('/kaggle/working/submission.csv',index=False)

