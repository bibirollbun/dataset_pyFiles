import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import hyperopt
import time
from hyperopt import hp, fmin, tpe, Trials, partial, STATUS_OK
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv')
test = pd.read_csv('/kaggle/input/london-house-price-prediction-advanced-techniques/test.csv')


train.head()


train.info()


for i in train.columns:
    print(train[i].value_counts())


for i in test.columns:
    print(test[i].value_counts())


train.isnull().sum()


train['bathrooms'] = train['bathrooms'].fillna(0)


train['bedrooms'] = train['bedrooms'].fillna(0)


train['livingRooms'] = train['livingRooms'].fillna(0)


train['tenure'] = train['tenure'].fillna('Unknown')


train['propertyType'] = train['propertyType'].fillna('Unknown')


train['currentEnergyRating'] = train['currentEnergyRating'].fillna('Unknown')


train['floorAreaSqM'] = train['floorAreaSqM'].fillna(train['floorAreaSqM'].mean())


del train['fullAddress']
del train['postcode']
del train['country']
del train['outcode']
del test['fullAddress']
del test['postcode']
del test['country']
del test['outcode']


train.isnull().sum()


test.isnull().sum()


test['bathrooms'] = test['bathrooms'].fillna(0)
test['bedrooms'] = test['bedrooms'].fillna(0)
test['livingRooms'] = test['livingRooms'].fillna(0)
test['tenure'] = test['tenure'].fillna('Unknown')
test['propertyType'] = test['propertyType'].fillna('Unknown')
test['currentEnergyRating'] = test['currentEnergyRating'].fillna('Unknown')
test['floorAreaSqM'] = test['floorAreaSqM'].fillna(train['floorAreaSqM'].mean())


test.isnull().sum()


train['sale_year']


train['sale_year'].value_counts()


train['sale_year'] = train['sale_year']-1995


train['sale_year'].value_counts()


test['sale_year'] = test['sale_year']-1995


def find_season(month):
        season_month_north = {
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Autumn', 10:'Autumn', 11:'Autumn'}
        return season_month_north.get(month)


season_list = []
for month in train['sale_month']:
    season = find_season(month)
    season_list.append(season)
    
train['Season'] = season_list


train['Season'].value_counts()


season_list = []
for month in test['sale_month']:
    season = find_season(month)
    season_list.append(season)
    
test['Season'] = season_list


test['Season'].value_counts()


cats = ['bathrooms','bedrooms','livingRooms','currentEnergyRating','tenure','propertyType','Season']
num_cols = ['latitude','longitude','floorAreaSqM','sale_year','sale_month']


train.shape[1]


len(cats)+len(num_cols)+2


encoder = OrdinalEncoder()
train[cats] = encoder.fit_transform(train[cats])
test[cats] = encoder.transform(test[cats])


train[cats]


scaler = StandardScaler()


train[num_cols] = scaler.fit_transform(train[num_cols].values)


test[num_cols] = scaler.transform(test[num_cols].values)


ID_col = 'ID'
target = 'price'


X_train = train.drop(columns=[ID_col, target]).copy()
X_test = test.drop(columns=[ID_col]).copy()
y_train = train['price'].copy()


data_xgb = xgb.DMatrix(X_train,label=y_train,enable_categorical = True)


#First-Round range of hyperparameters

param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",50,200,10)
                     ,"eta": hp.quniform("eta",0.05,2.05,0.05)
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.4,1,0.1)
                     ,"lambda":hp.quniform("lambda",0,3,0.2)
                     ,"min_child_weight":hp.quniform("min_child_weight",50,600,50)
                     ,"max_depth":hp.choice("max_depth",range(3,10))
                     ,"subsample":hp.quniform("subsample",0.5,1,0.1)
                     ,"rate_drop":hp.quniform("rate_drop",0.1,1,0.1)
                    }


def hyperopt_objective(params):
    paramsforxgb = {"eta":params["eta"]
                    ,"colsample_bytree":params["colsample_bytree"]
                    ,"lambda":params["lambda"]
                    ,"min_child_weight":params["min_child_weight"]
                    ,"max_depth":int(params["max_depth"])
                    ,"subsample":params["subsample"]
                    ,"rate_drop":params["rate_drop"]
                    ,"nthread":14
                    ,"verbosity":0
                    ,"seed":1412}
    result = xgb.cv(params,data_xgb, seed=1412, metrics=("mae")
                    ,num_boost_round=int(params["num_boost_round"]))
    return result.iloc[-1,2]


def param_hyperopt(max_evals=100):
    
    trials = Trials()
    
    early_stop_fn = no_progress_loss(30)
    
    params_best = fmin(hyperopt_objective
                       , space = param_grid_simple
                       , algo = tpe.suggest
                       , max_evals = max_evals
                       , verbose=True
                       , trials = trials
                       , early_stop_fn = early_stop_fn
                      )
    
    print("\n","\n","best params: ", params_best,
          "\n")
    return params_best, trials


#I used these four lines of code to search for hyperparameters and measure the running time. I ran it five times to obtain five sets of parameters.
#start = time.time()
#params_best, trials = param_hyperopt(100)
#end = time.time()
#print(end - start)


#Here is the output if you run the code,it prompt the best parameters and it takes 406.9175612926483 seconds.

#100%|██████████| 100/100 [06:46<00:00,  4.07s/trial, best loss: 186994.69839028255]

#best params:  {'colsample_bytree': 0.7000000000000001, 'eta': 0.1, 'lambda': 0.4, 'max_depth': 5, 'min_child_weight': 100.0, 'num_boost_round': 180.0, 'rate_drop': 1.0, 'subsample': 0.8} 

#406.9175612926483

# The hp.choice function outputs an index. For example, "max_depth": hp.choice("max_depth", range(3, 10)) means that if 'max_depth' is 5, it actually corresponds to the value 8.



#Second-Round range of hyperparameters
param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",100,500,50)
                     ,"eta": hp.quniform("eta",0.01,0.15,0.02)
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.6,1,0.1)
                     ,"lambda":hp.quniform("lambda",0,2,0.1)
                     ,"min_child_weight":hp.quniform("min_child_weight",50,250,25)
                     ,"max_depth":hp.choice("max_depth",range(6,10))
                     ,"subsample":hp.quniform("subsample",0.7,1,0.1)
                     ,"rate_drop":hp.quniform("rate_drop",0.1,1,0.1)
                    }


#I used these four lines of code to search for hyperparameters and measure the running time. I ran it five times to obtain five sets of parameters.
#start = time.time()
#params_best, trials = param_hyperopt(100)
#end = time.time()
#print(end - start)


#Third-Round range of hyperparameters
param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",500,2500,500)
                     ,"eta": hp.quniform("eta",0.01,0.1,0.005)
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.7,1,0.05)
                     ,"lambda":hp.quniform("lambda",0.8,2,0.1)
                     ,"min_child_weight":hp.quniform("min_child_weight",20,70,5)
                     ,"max_depth":hp.choice("max_depth",range(6,10))
                     ,"subsample":hp.quniform("subsample",0.8,1,0.1)
                     ,"rate_drop":hp.quniform("rate_drop",0.1,1,0.1)
                    }


#I used these four lines of code to search for hyperparameters and measure the running time. I ran it five times to obtain five sets of parameters.
#start = time.time()
#params_best, trials = param_hyperopt(100)
#end = time.time()
#print(end - start)


#Here is the Third round result.
#best params:  {'colsample_bytree': 0.8, 'eta': 0.02, 'lambda': 1.9000000000000001, 'max_depth': 3, 'min_child_weight': 30.0, 'num_boost_round': 500.0, 'rate_drop': 0.4, 'subsample': 0.9} 

#best params:  {'colsample_bytree': 0.9, 'eta': 0.015, 'lambda': 1.8, 'max_depth': 2, 'min_child_weight': 30.0, 'num_boost_round': 1000.0, 'rate_drop': 0.4, 'subsample': 1.0} 

#best params: {'colsample_bytree': 0.8500000000000001, 'eta': 0.01, 'lambda': 1.5, 'max_depth': 3, 'min_child_weight': 20.0, 'num_boost_round': 1000.0, 'rate_drop':0.30000000000000004, 'subsample': 0.8} 

#best params:  {'colsample_bytree': 0.8, 'eta': 0.025, 'lambda': 2.0, 'max_depth': 3, 'min_child_weight': 25.0, 'num_boost_round': 500.0, 'rate_drop': 0.5, 'subsample': 1.0} 

#best params:  {'colsample_bytree': 0.9500000000000001, 'eta': 0.015, 'lambda': 1.4000000000000001, 'max_depth': 3, 'min_child_weight': 20.0, 'num_boost_round': 1500.0, 'rate_drop': 0.1, 'subsample': 0.9} 



# 57%|█████▋    | 57/100 [26:48<20:13, 28.22s/trial, best loss: 170350.25443484596]  
 
#best params: {'colsample_bytree': 0.8, 'eta': 0.025, 'lambda': 2.0, 'max_depth': 3, 'min_child_weight': 25.0, 'num_boost_round': 500.0, 'rate_drop': 0.5, 'subsample': 1.0}

# Leaderboard Score 223218.504742

# We adjust the hyperparameter searching range according to the Third-round 


def objective(params):
    model = XGBRegressor(
        n_estimators=int(params['num_boost_round']),
        learning_rate=params['eta'],
        colsample_bytree=params['colsample_bytree'],
        reg_lambda=params['lambda'],
        min_child_weight=params['min_child_weight'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        rate_drop=params['rate_drop'],
        random_state=1412
    )
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_index, test_index in tscv.split(X_train):
        Xtrain, Xtest = X_train.iloc[train_index], X_train.iloc[test_index]
        ytrain, ytest = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(Xtrain, ytrain)
        predictions = model.predict(Xtest)
        score = mean_absolute_error(ytest, predictions)
        scores.append(score)

    return {'loss': np.mean(scores), 'status': STATUS_OK}

space = {'num_boost_round': hp.quniform("num_boost_round",500,2000,250)
                     ,"eta": hp.quniform("eta",0.005,0.03,0.005)
                     ,"colsample_bytree":hp.quniform("colsample_bytree",0.75,1,0.05)
                     ,"lambda":hp.quniform("lambda",0.8,2.5,0.2)
                     ,"min_child_weight":hp.quniform("min_child_weight",15,50,2)
                     ,"max_depth":hp.choice("max_depth",range(6,10))
                     ,"subsample":hp.quniform("subsample",0.8,1,0.1)
                     ,"rate_drop":hp.quniform("rate_drop",0.1,0.6,0.05)
                    }




#I used these three lines of code to search for hyperparameters and measure the running time
#trials = Trials()
#best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
#print("Best hyperparameters:", best)


model_xgb = XGBRegressor(num_boost_round =500,enable_categorical=True,colsample_bytree=0.8,eta=0.025,reg_lambda=2.0,max_depth=9,min_child_weight=25,rate_drop=0.5,subsample=1.0)


model_xgb.fit(X_train, y_train)


#encoder = OrdinalEncoder()
#X_test[cats] = encoder.fit_transform(X_test[cats])


pred_xgb = model_xgb.predict(X_test)


sub = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/sample_submission.csv")
sub.price = pred_xgb
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()


# Leaderboard Score 223218.504742





