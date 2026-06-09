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


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet,BayesianRidge,ARDRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from scipy.stats import reciprocal

warnings.filterwarnings('ignore',module="")


import optuna
from sklearn.model_selection import cross_val_score


Traindata=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
Testdata=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")



Traindata.dropna(inplace=True)
Testdata.dropna(inplace=True)


Traindata["date"] = pd.to_datetime(Traindata["date"])

Testdata["date"] = pd.to_datetime(Testdata["date"])


Traindata["date_day_of_year"]=Traindata["date"].dt.dayofyear.astype('float64')
Traindata["date_day_of_week"]=Traindata["date"].dt.dayofweek.astype('float64')
Traindata["date_day_of_month"]=Traindata["date"].dt.day.astype('float64')
Traindata["date_quarter"]=Traindata["date"].dt.quarter.astype('float64')
Traindata['is_weekend'] = (Traindata['date_day_of_week'] >= 5).astype(int)  





Testdata["date_day_of_year"]=Testdata["date"].dt.dayofyear.astype('float64')
Testdata["date_day_of_week"]=Testdata["date"].dt.dayofweek.astype('float64')
Testdata["date_day_of_month"]=Testdata["date"].dt.day.astype('float64')
Testdata["date_quarter"]=Testdata["date"].dt.quarter.astype('float64')
Testdata['is_weekend'] = (Testdata['date_day_of_week'] >= 5).astype(int)  



Traindata["date_year"]=Traindata["date"].dt.year.astype('float64')
Traindata["date_month"]=Traindata["date"].dt.month.astype('float64')
Traindata["date_week"]=Traindata["date"].dt.isocalendar().week.astype('float64')
Traindata['Group'] = (Traindata['date_year'] - 2010) * 48 + Traindata['date_month'] * 4 + Traindata['date_day_of_month'] // 7




Testdata["date_year"]=Testdata["date"].dt.year.astype('float64')
Testdata["date_month"]=Testdata["date"].dt.month.astype('float64')
Testdata["date_week"]=Testdata["date"].dt.isocalendar().week.astype('float64')
Testdata['Group'] = (Testdata['date_year'] - 2010) * 48 + Testdata['date_month'] * 4 + Testdata['date_day_of_month'] // 7


Traindata["week_sine"]=np.sin(2 * np.pi * Traindata["date_week"] / 52.0)
Traindata["month_sine"]=np.sin(2 * np.pi * Traindata["date_month"] /12.0)
Traindata["year_sine"]=np.sin(2 * np.pi * Traindata["date_year"] / 7.0)
Traindata["day_of_year_sine"]=np.sin(2 * np.pi * Traindata["date_day_of_year"] / 365.0)
Traindata["day_of_month_sine"]=np.sin(2 * np.pi * Traindata["date_day_of_month"] / 30.5)
Traindata["quarter_sine"]=np.sin(2 * np.pi * Traindata["date_quarter"] / 4.0)




Testdata["week_sine"]=np.sin(2 * np.pi * Testdata["date_week"] / 52.0)
Testdata["month_sine"]=np.sin(2 * np.pi * Testdata["date_month"] /12.0)
Testdata["year_sine"]=np.sin(2 * np.pi * Testdata["date_year"] / 7.0)
Testdata["day_of_year_sine"]=np.sin(2 * np.pi * Testdata["date_day_of_year"] / 365.0)
Testdata["day_of_month_sine"]=np.sin(2 * np.pi * Testdata["date_day_of_month"] / 30.5)
Testdata["quarter_sine"]=np.sin(2 * np.pi * Testdata["date_quarter"] / 4.0)


Traindata["week_cos"]=np.cos(2 * np.pi * Traindata["date_week"] / 52.0)
Traindata["month_cos"]=np.cos(2 * np.pi * Traindata["date_month"] /12.0)
Traindata["year_cos"]=np.cos(2 * np.pi * Traindata["date_year"] / 7.0)
Traindata["day_of_year_cos"]=np.cos(2 * np.pi * Traindata["date_day_of_year"] / 365.0)
Traindata["day_of_month_cos"]=np.cos(2 * np.pi * Traindata["date_day_of_month"] / 30.5)
Traindata["quarter_cos"]=np.cos(2 * np.pi * Traindata["date_quarter"] / 4.0)




Testdata["week_cos"]=np.cos(2 * np.pi * Testdata["date_week"] / 52.0)
Testdata["month_cos"]=np.cos(2 * np.pi * Testdata["date_month"] /12.0)
Testdata["year_cos"]=np.cos(2 * np.pi * Testdata["date_year"] / 7.0)
Testdata["day_of_year_cos"]=np.cos(2 * np.pi * Testdata["date_day_of_year"] / 365.0)
Testdata["day_of_month_cos"]=np.cos(2 * np.pi * Testdata["date_day_of_month"] / 30.5)
Testdata["quarter_cos"]=np.cos(2 * np.pi * Testdata["date_quarter"] / 4.0)


TrainColumn=['country', 'store', 'product', 'week_sine', 'month_sine',
       'year_sine', 'day_of_year_sine', 'day_of_month_sine', 'week_cos',
       'month_cos', 'year_cos', 'day_of_year_cos', 'day_of_month_cos','quarter_cos','quarter_sine','Group','date_day_of_month','date_quarter',
            'date_month','date_week','is_weekend','date_year','date_day_of_year','date_day_of_week']


Traindata2=Traindata[TrainColumn]
Testdata2=Testdata[TrainColumn]
Ytrain=np.log(Traindata["num_sold"])


cat_columns=["country","store","product"]



Testdata2=pd.get_dummies(Testdata2,columns=cat_columns,drop_first=True)
Traindata2=pd.get_dummies(Traindata2,columns=cat_columns,drop_first=True)



TrainColumnFinal=['week_sine', 'month_sine', 'year_sine',
       'day_of_year_sine', 'day_of_month_sine', 'week_cos', 'month_cos',
       'year_cos', 'day_of_year_cos', 'day_of_month_cos', 'quarter_cos',
       'quarter_sine', 'country_Finland', 'country_Italy', 'country_Kenya',
       'country_Norway', 'country_Singapore', 'store_Premium Sticker Mart',
       'store_Stickers for Less', 'product_Kaggle', 'product_Kaggle Tiers',
       'product_Kerneler', 'product_Kerneler Dark Mode','Group','date_day_of_month','date_quarter',
            'date_month','date_week','is_weekend','date_day_of_year','date_day_of_week']


TraindataFinal=Traindata2[TrainColumnFinal]
TestdataFinal=Testdata2[TrainColumnFinal]


# X_train, X_test, y_train, y_test=train_test_split(TraindataFinal,Ytrain, test_size=0.05,shuffle=True)


def objective_lgbm(trial):
    lgbm_params = {
        "n_estimators": trial.suggest_int('n_estimators', 700, 2000) ,
        # "n_estimators": 100 ,

        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 60, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 0.5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)

    }
    model = LGBMRegressor(**lgbm_params,verbose=-1,random_state=69) 
    X_train, X_test, y_train, y_test=train_test_split(TraindataFinal,Ytrain, test_size=0.05,shuffle=True)
    # score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring='neg_mean_absolute_percentage_error').mean()


    model.fit(X_train, y_train)
    y_pred = np.exp(model.predict(X_test))
    return mean_absolute_percentage_error(np.exp(y_test), y_pred)


def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 700, 1600)
    learning_rate= trial.suggest_loguniform('learning_rate', 0.01, 1),  # Learning rate

    max_depth = trial.suggest_categorical('max_depth',[4,8,10,12,14,15,16,17,18,20,22,24] )
    reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10, log=True)
    subsample=trial.suggest_categorical("subsample",[0.5,0.6,0.7,0.8,0.9,1])
    # booster = trial.suggest_categorical('booster', ['gbtree','dart']) 
    gamma=trial.suggest_float('gamma', 1e-2, 12,step=0.1) 
    # colsample_bytree= trial.suggest_uniform('colsample_bytree', 0.6, 1.0),  # Feature subsampling

    
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, reg_lambda=reg_lambda,subsample=subsample,gamma=gamma,random_state=69) 

    # score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring='neg_mean_absolute_percentage_error').mean()
    model.fit(X_train, y_train)
    y_pred = np.exp(model.predict(X_test))
    return mean_absolute_percentage_error(np.exp(y_test), y_pred)
    # return score


def model_configs(model_name):
    if model_name=="xgb":
        return objective_xgboost
    elif model_name=="lgbm":
        return objective_lgbm
    elif model_name=="rf":
        return objective_Rf


def Training(model_list):
    # model_dict={}
    # model_dict[store]=[]
    result={}
    for model_name in model_list:
        print("Current model is :- ",model_name)
        objective=model_configs(model_name)
        study = optuna.create_study(direction='minimize') 
        study.optimize(objective, n_trials=60,show_progress_bar=True)
        best_params = study.best_params
        # result.append((model_name,best_params))
        result[model_name]=best_params

        
        

        # model_dict[store].append((model_name,best_model))
    return result


modelList=["lgbm"]


optuna.logging.set_verbosity(optuna.logging.WARNING)



res=Training(modelList)


best_model = LGBMRegressor(**res["lgbm"],verbose=-1,random_state=69) 
best_model.fit(TraindataFinal, Ytrain)


testPreds=best_model.predict(TestdataFinal)


result=pd.DataFrame({'id': Testdata['id'], 'target': np.exp(testPreds)})



result.to_csv("/kaggle/working/submission.csv",index=False)




