import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline


dataTrain = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
dataTest = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


data = pd.concat([dataTrain, dataTest], axis=0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)


data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
#週数,日を追加
data['yearweek'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").isocalendar().week)
data['day'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").day)


from sklearn.ensemble import RandomForestRegressor

dataWind0 = data[data["windspeed"]==0]
dataWindNot0 = data[data["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = pd.concat([dataWindNot0, dataWind0], axis=0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)


# humidity=0の値を消去
data.drop(data[data['humidity'] == 0].index, inplace=True)


#特徴量に週数を追加
categoricalFeatureNames = ["season","holiday","workingday","weather","yearweek","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered","day"]


for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")


dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

dataTrain_x = dataTrain[dataTrain['day'] <= 15]
dataTrain_y = dataTrain[dataTrain['day'] > 15]

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]
yLabelsRegistered = dataTrain["registered"]
yLabelsCasual = dataTrain["casual"]

yLabels_x = dataTrain_x["count"]
yLabelsRegistered_x = dataTrain_x["registered"]
yLabelsCasual_x = dataTrain_x["casual"]

yLabels_y = dataTrain_y["count"]
yLabelsRegistered_y = dataTrain_y["registered"]
yLabelsCasual_y = dataTrain_y["casual"]


dataTrain = dataTrain.drop(dropFeatures,axis=1)
dataTrain_x = dataTrain_x.drop(dropFeatures,axis=1)
dataTrain_y = dataTrain_y.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)


def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# ランダムフォレストモデルの構築
from sklearn.ensemble import RandomForestRegressor
rfModel_R = RandomForestRegressor(n_estimators = 5000)
yLabelsLogRegistered = np.log1p(yLabelsRegistered)
rfModel_R.fit(dataTrain,yLabelsLogRegistered)
rf_preds_R = rfModel_R.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.expm1(yLabelsLogRegistered),np.expm1(rf_preds_R),False))

# ランダムフォレストモデルでの予測
rf_predsTest_R = rfModel_R.predict(X= dataTest)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sn.distplot(yLabelsRegistered,ax=ax1,bins=50)
sn.distplot(np.expm1(rf_predsTest_R),ax=ax2,bins=50)


# ランダムフォレストモデルの構築
from sklearn.ensemble import RandomForestRegressor
rfModel_C = RandomForestRegressor(n_estimators = 5000)
yLabelsLogCasual = np.log1p(yLabelsCasual)
rfModel_C.fit(dataTrain,yLabelsLogCasual)
rf_preds_C = rfModel_C.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.expm1(yLabelsLogCasual),np.expm1(rf_preds_C),False))

# ランダムフォレストモデルでの予測
rf_predsTest_C = rfModel_C.predict(X= dataTest)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sn.distplot(yLabelsCasual,ax=ax1,bins=50)
sn.distplot(np.expm1(rf_predsTest_C),ax=ax2,bins=50)


import optuna
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
yLabelsLogRegistered_x = np.log1p(yLabelsRegistered_x)
yLabelsLogRegistered_y = np.log1p(yLabelsRegistered_y)

def objective_R(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "verbosity": -1
    }

    LGBM = LGBMRegressor(**params, n_estimators=5000)
    LGBM.fit(dataTrain_x,yLabelsLogRegistered_x,
              eval_set=[(dataTrain_y,yLabelsLogRegistered_y)],
              eval_metric="rmse",
              callbacks=[
                  early_stopping(stopping_rounds=200),
                  log_evaluation(0) 
              ])

    preds = LGBM.predict(dataTrain_y)
    rmse = mean_squared_error(yLabelsLogRegistered_y, preds, squared=False)
    return rmse

study_R = optuna.create_study(direction="minimize")
study_R.optimize(objective_R, n_trials=200)

print("Best Score:", study_R.best_value)
print("Best Params:", study_R.best_params)


from lightgbm import LGBMRegressor
params_R = study_R.best_params
LGBMR_R = LGBMRegressor(**params_R, n_estimators=5000)
yLabelsLogRegistered = np.log1p(yLabelsRegistered)
LGBMR_R.fit(dataTrain,yLabelsLogRegistered)
LGBMR_preds_R = LGBMR_R.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.expm1(yLabelsLogRegistered),np.expm1(LGBMR_preds_R),False))

LGBMR_predsTest_R = LGBMR_R.predict(X= dataTest)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sn.distplot(yLabelsRegistered,ax=ax1,bins=50)
sn.distplot(np.expm1(LGBMR_predsTest_R),ax=ax2,bins=50)


import optuna
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
yLabelsLogCasual_x = np.log1p(yLabelsCasual_x)
yLabelsLogCasual_y = np.log1p(yLabelsCasual_y)

def objective_C(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "verbosity": -1
    }

    LGBM = LGBMRegressor(**params, n_estimators=5000)
    LGBM.fit(dataTrain_x,yLabelsLogCasual_x,
              eval_set=[(dataTrain_y,yLabelsLogCasual_y)],
              eval_metric="rmse",
              callbacks=[
                  early_stopping(stopping_rounds=200),
                  log_evaluation(0) 
              ])

    preds = LGBM.predict(dataTrain_y)
    rmse = mean_squared_error(yLabelsLogCasual_y, preds, squared=False)
    return rmse

study_C = optuna.create_study(direction="minimize")
study_C.optimize(objective_C, n_trials=200)

print("Best Score:", study_C.best_value)
print("Best Params:", study_C.best_params)


from lightgbm import LGBMRegressor
params_C = study_C.best_params
LGBMR_C = LGBMRegressor(**params_C, n_estimators=5000)
yLabelsLogCasual = np.log1p(yLabelsCasual)
LGBMR_C.fit(dataTrain,yLabelsLogCasual)
LGBMR_preds_C = LGBMR_C.predict(X= dataTrain)
print ("RMSLE Value For Gradient Boost: ",rmsle(np.expm1(yLabelsLogCasual),np.expm1(LGBMR_preds_C),False))

LGBMR_predsTest_C = LGBMR_C.predict(X= dataTest)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sn.distplot(yLabelsCasual,ax=ax1,bins=50)
sn.distplot(np.expm1(LGBMR_predsTest_C),ax=ax2,bins=50)


submission = pd.DataFrame({
        "datetime": datetimecol,
        "count": [max(0, x) for x in 0.8*(np.expm1(LGBMR_predsTest_R) + np.expm1(LGBMR_predsTest_C)) + 0.2*(np.expm1(rf_predsTest_R) + np.expm1(rf_predsTest_C))]
    })
submission.to_csv('output_K_O.csv', index=False)

