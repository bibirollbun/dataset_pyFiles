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



#Check original dataset with adversarial validation.


###############
### general ###
###############
import pandas as pd
import numpy as np
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import gc
import time
###############
### metrics ###
###############
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import skew
from scipy import stats
##################
### processing ###
##################
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelmin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import minmax_scaling
from sklearn.utils import shuffle
from eli5.sklearn import PermutationImportance
import eli5

##############
### models ###
##############
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,AdaBoostClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier,IsolationForest, VotingClassifier,ExtraTreesRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.svm import (SVC,SVR)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import (LogisticRegression,LinearRegression,LassoCV)
from sklearn.linear_model import RidgeCV
#from sklearn.neighbors import (KNeighborsClassifier,KNeighborsRegressor)
from catboost import (CatBoostRegressor,CatBoostClassifier)
#from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb


######################
### neural network ###
######################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras import utils
import keras_tuner
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters, Objective
from scipy import stats
import optuna
##########
### CV ###
##########
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test.drop("id",axis=1,inplace=True)
test['adv_val'] =  0

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train.drop("id",axis=1,inplace=True)
train['adv_val'] =  1

original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
original['adv_val'] =  2

sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


sets={'train':train,'test':test,'original':original}


temp=pd.DataFrame()
for q in range(7):
    a=pd.DataFrame(train.describe().iloc[q]).T
    b=pd.DataFrame(test.describe().iloc[q]).T
    c=pd.DataFrame(original.describe().iloc[q]).T
    d=pd.concat([a,b,c],keys=['train', 'test','original'])
    temp=pd.concat([temp,d])
temp


train.info()
test.info()
original.info()


########################
### General Settings ###
########################

gpu_switch = False

if gpu_switch:
    method_LGBM = "gpu"
    method_XGB = "cuda"
    method_CAT = "GPU"
else:
    method_LGBM = "cpu"
    method_XGB = "cpu" 
    method_CAT = "CPU"

##########
### CV ###
##########

n_splits = 7
n_repeats =1
#cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats = n_repeats, random_state=2023)
#cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv = RepeatedKFold(n_splits=n_splits, n_repeats = n_repeats, random_state=2024)
#cv = GroupKFold(n_splits=n_splits)

########################
### Define Weights   ###
########################

#weights = {0: 0.5009553158705701, 1: 262.19354838709677}

#4 DROPPING DUPLICATES
drop_dup=True

#4.1 ENCODING AND IMPUTING
target='BeatsPerMinute'

#4.2 VISUALIZATION
vis_hist_active=False
vis_boxplot_active=False
vis_scatter_active=False
vis_scatter_exp_active=False

#4.2 ADVERSARIAL VALIDATION
adv_val=False
concat_with_orig=False
adv_val_perm_active=False

#4.3 PERMUTATION_IMPORTANCE
perm_switch=False

#4.4 OUTLIERS

#4.4.1 IQR
IQR_tuning_switch=False
IQR_apply_switch=True

#4.4.2 IF
IR_tuning_switch=False

#4.5 NORMALIZATION AND SCALING
normalization=False

#5 MODELING #6 ENSEMBLING

#######################
### Optuna Settings ###
#######################
direction = "minimize"
optuna_study = True 
optuna_models={'LGBM' : False, 
               'XGB'  : False,
               'CAT'  : False,
               'AB'   : False,
               'GB'   : False,
               'ET'   : False,
               'RF'   : False,
               'LCV'  : False,
               'LR'   : False,
               'KNC'  : False,
               'SVC'  : False,
               'KERAS': False}

#######################
### Finish Settings ###
#######################
finish_set = True
finish_models={'LGBM' : True,
               'XGB'  : True,
               'CAT'  : True,
               'AB'   : False,
               'GB'   : False,
               'ET'   : False,
               'RF'   : False,
               'LCV'  : False,
               'LR'   : False,
               'KNC'  : False,
               'SVC'  : False,
               'KERAS': False,
               'AUTOKERAS'  : False,
               'H2O'  : False,
               'FLAML'  : True}
##################
### def metric ###
##################
def metric_call():
    #k =roc_auc_score(valid_y,model.predict_proba(valid_X)[:,1])
    #k =accuracy_score(valid_y,model.predict(valid_X))
    k=mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
    return k

metric = 'Root Mean Squared Error'

LGBM_metrics= {'AUC/ROC' : 'auc',
               'Accuracy score'  : 'binary_logloss',
               'Root Mean Squared Error':'rmse'}

XGB_metrics=  {'AUC/ROC' : 'auc',
               'Accuracy score'  : 'logloss',
               'Root Mean Squared Error':'rmse'}

CAT_metrics=  {'AUC/ROC' : 'AUC',
               'Accuracy score': 'Accuracy',
               'Root Mean Squared Error':'RMSE'}

FLAML_metrics=  {'AUC/ROC' : 'roc_auc',
               'Accuracy score': 'accuracy',
               'Root Mean Squared Error':'rmse'}

Perm_metrics=  {'Root Mean Squared Error':'neg_mean_squared_error'}

#####################
### loss function ###
#####################
task = 'Regression'

LGBM_tasks= {'Regression' : '!!!'}

XGB_tasks=  {'Regression' : 'reg:squarederror'}

CAT_tasks=  {'Regression' : 'RMSE'}

FLAML_tasks=  {'Regression' : 'regression'}


###############################################
### The function for step-by-step analyzing ###
###############################################
param_FLAML_get_score = {'task':FLAML_tasks[task],
                          'n_jobs':4,
                          'metric':FLAML_metrics[metric],
                          'time_budget':120,
                          'verbose':3,
                          'ensemble':False,
                          'eval_method' : "cv",
                         'n_splits' :5,
                          'use_spark':True,
                        }
def FLAML_get_score(X,y):
    start_time = time.time()
    model = FlamlAutoML(**param_FLAML_get_score)
    model.fit(X,y) 
    result=mean_squared_error(y,model.predict(X))**(1/2)
    print (f'\033[0;33;40m Result = {result} \033[0;30;0m')

    print (model.best_result)


###############################################
### The function for step-by-step analyzing ###
###############################################
param_LGBM_get_score = {'metric': LGBM_metrics[metric], 
                        'early_stopping_round': 300, 
                        'n_estimators': 2000,
                        'learning_rate': 0.005,
                        'n_jobs':4,
                        'device': method_LGBM}
def LGBM_get_score(X,y):
    start_time = time.time()
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = lgb.LGBMRegressor(**param_LGBM_get_score).fit(train_X,train_y,
                                            eval_set=[(valid_X,valid_y)],
                                            callbacks=[lgb.log_evaluation(period=200, show_stdv=True)]) 
        result=mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        print (f'\033[0;33;40m Step#{i} Auc result = {result} \033[0;30;0m')
    print (f'\033[0;35;40m Final LGBM Result = {sum(results)/len(results)} \033[0;30;0m')
    print (f'time is {np.round((time.time()-start_time),1)}')


###############################################
### The function for step-by-step analyzing ###
###############################################

param_XGB_get_score = { 'device':method_XGB,
                        'eval_metric': XGB_metrics[metric],
                        'learning_rate': 0.001,
                        'n_estimators': 2000,
                        'early_stopping_rounds':200}

def XGB_get_score(X,y):
    start_time = time.time()
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = xgb.XGBRegressor(**param_XGB_get_score).fit(train_X,train_y,
                                                  eval_set=[(valid_X,valid_y)],
                                                  verbose=0)  
        result=mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        print (f'\033[0;33;40m Step#{i} result = {result} \033[0;30;0m')
    print (f'\033[0;35;40m Final CAT Result = {sum(results)/len(results)} \033[0;30;0m')
    print (f'time is {np.round((time.time()-start_time),1)}')


###############################################
### The function for step-by-step analyzing ###
###############################################
def RF_get_score(X,y):
    start_time = time.time()
    param = {'n_jobs':-1}
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestRegressor(**param).fit(train_X,train_y) 
        result=mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        print (f'\033[0;33;40m Step#{i} Auc result = {result} \033[0;30;0m')
    print (f'\033[0;35;40m Final LGBM Result = {sum(results)/len(results)} \033[0;30;0m')
    print (f'time is {np.round((time.time()-start_time),1)}')


#########################################################################
### The function for data distribution analysis between two datasets  ###
#########################################################################

def viz_comp (data1,data2,title):
    n_bins = 30
    histplot_hyperparams = {
        'kde':True,
        'alpha':0.2,
        'stat':'percent',
        'bins':n_bins,
        #'log_scale':(False, True)
    }
    cols=num_cols
    fig, ax = plt.subplots(len(cols),2, figsize=(20, 50))

    for i, column in enumerate(cols):
        sns.histplot(
            data1[column], label='Data1',
            ax=ax[i][0], color='green', **histplot_hyperparams)
        sns.histplot(
            data2[column], label='Data2',
            ax=ax[i][1], color='red', **histplot_hyperparams)
    #ax[0].set_title(title, fontstyle='normal',size=25)


#################
### Boxplots  ###
#################

def viz_boxplot (data1):
    cols=num_cols
    fig, ax = plt.subplots(len(cols)//3+1,3, figsize=(20, 20))
    ax = ax.flatten()
    title ='boxplot'
    for i, column in enumerate(cols):
        ax[i].boxplot(data1[column],0, 'gD')
        ax[i].set_title(column)

    ax[0].set_title(title, fontstyle='normal',size=25)


##############################
### Permutation importance ###
##############################

def permutation_importance(model,X,y):
    permute = PermutationImportance(model,random_state=2023,n_iter =10,cv=5).fit(X, y)
    eli5.show_weights(permute, feature_names = X.columns.tolist(),top=50)
    values = dict(zip(list(train.columns),list(permute.feature_importances_)))
    sorted_dict = {}
    sorted_keys = sorted(values, key=values.get)
    for w in sorted_keys:
        sorted_dict[w] = values[w]


def miss_values_check(data,n):
    print(f'\033[0;33;40m A number of NaN values in {n} is {data.isnull().sum().sum()} \033[0;30;0m')
    if data.isnull().sum().sum() >0:
        sns.heatmap(data.isnull())
for n in sets:
    miss_values_check(sets[n],n)


drop_duplcated_list={'train':train,'original':original}

def dropping_duplicates(data):
    data.drop_duplicates(inplace = True)

for n in drop_duplcated_list:
    print(f"\033[0;33;40m A number of duplicated rows in {n} is {sets[n].duplicated().sum()}, they were dropped \033[0;30;0m")
    dropping_duplicates(sets[n])
    drop_duplcated_list[n].reset_index(drop=True,inplace=True)


for x in train.columns:
    print(f'\033[0;33;40m Unique values in train {x} = \033[0;35;40m {len(train[x].unique())}  ')


cat_cols = []
drop_col = []
remain_col =['BeatsPerMinute']
num_cols=['RhythmScore','AudioLoudness','VocalContent','AcousticQuality','InstrumentalScore',	
'LivePerformanceLikelihood','MoodScore','TrackDurationMs','Energy']



train.drop(drop_col,axis=1,inplace=True)
test.drop(drop_col,axis=1,inplace=True)
original.drop(drop_col,axis=1,inplace=True)


num_imp = SimpleImputer(strategy='mean')
cat_imp = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(handle_unknown='ignore',sparse = False,
                    #drop="first"
                   )
#ohe = OrdinalEncoder()


train[num_cols] = pd.DataFrame(num_imp.fit_transform(train[num_cols]),columns=num_cols)
#train[cat_cols] = pd.DataFrame(cat_imp.fit_transform(train[cat_cols]),columns=cat_cols)
temp=pd.DataFrame(ohe.fit_transform(train[cat_cols]),columns=ohe.get_feature_names_out())
train=pd.concat([train.drop(cat_cols,axis=1),temp],axis=1) 

original[num_cols] = pd.DataFrame(num_imp.transform(original[num_cols]),columns=num_cols)
#original[cat_cols] = pd.DataFrame(cat_imp.transform(original[cat_cols]),columns=cat_cols)
temp=pd.DataFrame(ohe.transform(original[cat_cols]),columns=ohe.get_feature_names_out())
original=pd.concat([original.drop(cat_cols,axis=1),temp],axis=1)    

test[num_cols] = pd.DataFrame(num_imp.transform(test[num_cols]),columns=num_cols)
#test[cat_cols] = pd.DataFrame(cat_imp.transform(test[cat_cols]),columns=cat_cols)
temp=pd.DataFrame(ohe.transform(test[cat_cols]),columns=ohe.get_feature_names_out())
test=pd.concat([test.drop(cat_cols,axis=1),temp],axis=1)


X = train.drop([target],axis=1)
X_orig=original.drop([target],axis=1)

y = train[target]
y_orig=original[target]

X,y=shuffle(X,y,random_state=2024)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

X_orig,y_orig=shuffle(X_orig,y_orig,random_state=2024)
X_orig = X_orig.reset_index(drop=True)
y_orig = y_orig.reset_index(drop=True)


########################
### Define Weights   ###
########################
classes = y.unique()
weight = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
weights = dict(zip(classes, list(weight)))


sns.heatmap(train.corr(),annot=True,fmt=".2f")


sns.heatmap(test.corr(),annot=True,fmt=".2f")


if vis_hist_active: #general_settings 4.2
    viz_comp(X,test,"X vs TEST")
    viz_comp(X_orig,test,"X_orig vs TEST")


if vis_boxplot_active: #general_settings 4.2
    viz_boxplot(X)
    viz_boxplot(X_orig)
    viz_boxplot(test)


if vis_scatter_active:
    fig, ax = plt.subplots(3,3, figsize=(20, 20))
    ax = ax.flatten()
    i=0
    for i,n in enumerate(list(train.columns)[:-2]):
        ax[i].scatter(x=(train[n]*1000),y=(train[target]),s=0.1,label=n)
        ax[i].set_title(f"{n}+{target}")


if vis_scatter_exp_active:
    fig, ax = plt.subplots(27,3, figsize=(20, 160))
    ax = ax.flatten()
    i=0
    for i,n in enumerate(list(train.columns)[:-2]):
        for z,m in enumerate(list(train.columns)[:-2]):
            ax[i*9+z].scatter(x=(train[n]*train[m])**(6),y=train[target],s=0.1)
            ax[i*9+z].set_title(f'{n}*{m}+{target}')


def adv_validation (data_1,data_2):
    X_temp=shuffle(pd.concat([data_1,data_2],ignore_index=True))  
    X_full=X_temp.drop(['adv_val'],axis=1)
    y_full=X_temp['adv_val']
    X_full,y_full=shuffle(X_full,y_full)
    X_full=X_full.reset_index(drop=True)
    y_full=y_full.reset_index(drop=True)
    result = cross_val_score(lgb.LGBMClassifier(n_estimators=200,verbose=-1),X_full,y_full,scoring='roc_auc',cv=5).mean()
    return result


if adv_val: #general_settings 4.2
    sets_adv_val={'train':X,'original':X_orig}
    for one in sets_adv_val:
        print (f'\033[0;33;40m Result between {one} and test = {adv_validation(sets_adv_val[one],test)} \033[0;30;0m')


if adv_val_perm_active: #general_settings 4.2
    model = lgb.LGBMClassifier(verbose=-1)
    metric='roc_auc'
    X_orig_test_perm=pd.concat([X_orig,test]).drop(['adv_val'],axis=1).reset_index(drop=True)
    y_orig_test_perm=pd.concat([X_orig,test])['adv_val'].reset_index(drop=True)
    permute = PermutationImportance(model,random_state=2024,n_iter =1,cv=5,scoring = metric).fit(X_orig_test_perm, y_orig_test_perm)
    eli5.show_weights(permute, feature_names = X_orig_test_perm.columns.tolist(),top=50)


X = X.drop(['adv_val'],axis=1)
X_orig = X_orig.drop(['adv_val'],axis=1)
test = test.drop(['adv_val'],axis=1)


if concat_with_orig: #general_settings 4.2
    X=pd.concat([X,X_orig],axis=0).reset_index(drop=True)
    y=pd.concat([y,y_orig],axis=0).reset_index(drop=True)


def perm_imp(model,data,target):
    X = data.to_numpy().copy()
    y = target.to_numpy().copy()
    permute = PermutationImportance(model,random_state=2023,n_iter =1,cv=5,scoring='roc_auc').fit(X, y)
    eli5.show_weights(permute, feature_names = data.columns.tolist(),top=50)
    values = dict(zip(list(data.columns),list(permute.feature_importances_)))
    sorted_dict = {}
    sorted_keys = sorted(values, key=values.get)
    for w in sorted_keys:
        sorted_dict[w] = np.round(values[w],3)
    return sorted_dict


if perm_switch:
    model = lgb.LGBMRegressor(n_estimators=1000,verbose=-1)
    #X=shuffle(pd.concat([original,test],ignore_index=True).drop(['adv_val',target],axis=1),random_state=2023).reset_index(drop=True)
    #y=shuffle(pd.concat([original,test],ignore_index=True).adv_val,random_state=2023).reset_index(drop=True)
    permute = PermutationImportance(model,random_state=2024,n_iter =1,cv=5,scoring = Perm_metrics[metric]).fit(X, y)
    eli5.show_weights(permute, feature_names = X.columns.tolist(),top=100)


X['AudioLoudness']=-X['AudioLoudness']
test['AudioLoudness']=-test['AudioLoudness']
X['TrackDurationMs']=X['TrackDurationMs']/(60*60*60)
test['TrackDurationMs']=test['TrackDurationMs']/(60*60*60)


def Box_transform(X,test):
    box_cols = num_cols
    for column in box_cols: 
        X_temp,fitted_lambda = stats.boxcox(X[column]) 
        X[column]=X_temp 
        test_temp = stats.boxcox(test[column],fitted_lambda) 
        test[column]=test_temp


Box_transform(X,test)


def StaSca_transform(X,test):
    StaSca = StandardScaler()
    X[num_cols] = pd.DataFrame(data = StaSca.fit_transform(X[num_cols]),columns = X[num_cols].columns)
    test[num_cols] = pd.DataFrame(data = StaSca.transform(test[num_cols]),columns = test[num_cols].columns)


if IQR_tuning_switch:
    def objective(trial):
        IQR_k_features={}
        for i,n in enumerate(list(X.columns)):
            IQR_k_features["k_"+n]=[trial.suggest_float("k_left_"+n, 0.0, 3.0,step=0.05),
                                    trial.suggest_float("k_right_"+n, 0.0, 3.0,step=0.05)]
        indexies_all=[]
        for n in IQR_k_features:
            k=n.replace("k_","")
            q1=X[k].quantile(0.25)
            q3=X[k].quantile(0.75)
            iqr=q3-q1
            max_value=q3+iqr*IQR_k_features[n][1]
            min_value=q1-iqr*IQR_k_features[n][0]
            indexies = X[(X[k]>max_value) | (X[k]<min_value)].index
            indexies_all.extend(indexies)
            
        results=[]
        n_iterations=[]
        for train_index, test_index in cv.split(X, y):
            train_X, valid_X = X.loc[train_index].drop(indexies_all,axis=0,errors="ignore"), X.loc[test_index]
            train_y, valid_y = y.loc[train_index].drop(indexies_all,axis=0,errors="ignore"), y.loc[test_index]
           
            model = xgb.XGBRegressor(**param_XGB_get_score).fit(train_X,train_y,
                                          eval_set=[(valid_X,valid_y)],
                                          verbose=0)  
            result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
            results.append(result)
            print(result)
        n=sum(results)/len(results)   
        return n
    
    
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                        direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


IQR_k_params={  'k_left_RhythmScore': 2.0500000000000003, 
                'k_right_RhythmScore': 1.0, 
                'k_left_AudioLoudness': 2.45, 
                'k_right_AudioLoudness': 1.25, 
                'k_left_VocalContent': 1.6500000000000001, 
                'k_right_VocalContent': 2.0500000000000003, 
                'k_left_AcousticQuality': 1.6500000000000001, 
                'k_right_AcousticQuality': 2.5500000000000003, 
                'k_left_InstrumentalScore': 1.1500000000000001, 
                'k_right_InstrumentalScore': 2.9000000000000004, 
                'k_left_LivePerformanceLikelihood': 1.6, 
                'k_right_LivePerformanceLikelihood': 1.4500000000000002, 
                'k_left_MoodScore': 1.55, 
                'k_right_MoodScore': 2.25, 
                'k_left_TrackDurationMs': 1.6500000000000001, 
                'k_right_TrackDurationMs': 0.9500000000000001, 
                'k_left_Energy': 2.2, 
                'k_right_Energy': 1.75} 
#Best is trial 248 with value: 26.45984922377388.


IQR_k_features={}
for i,n in enumerate(num_cols):
    IQR_k_features[n]=[list(IQR_k_params.values())[i*2],list(IQR_k_params.values())[i*2+1]]


if IQR_apply_switch:
    indexies_all=[]
    for n in IQR_k_features:
        k=n.replace("k_","")
        q1=X[k].quantile(0.25)
        q3=X[k].quantile(0.75)
        iqr=q3-q1
        max_value=q3+iqr*IQR_k_features[n][1]
        min_value=q1-iqr*IQR_k_features[n][0]
        indexies = X[(X[k]>max_value) | (X[k]<min_value)].index
        indexies_all.extend(indexies)
    X=X.drop(indexies_all,axis=0,errors="ignore").reset_index(drop=True)
    y=y.drop(indexies_all,axis=0,errors="ignore").reset_index(drop=True)


if IR_tuning_switch:
    def objective(trial):
        IR_k_contamination=trial.suggest_float("k_contamination", 0.0001, 0.0020,step=0.00001)
        IR_df_temp=pd.DataFrame()
        model = IsolationForest(n_estimators=200, 
                                max_samples=500, 
                                contamination=IR_k_contamination, 
                                max_features=2, 
                                random_state=2025,
                                verbose=1,
                                n_jobs=-4)
        model.fit(X)
        IR_df_temp['is_anomaly'] = model.predict(X)
        IR_df_temp['is_anomaly'] = IR_df_temp['is_anomaly'].map({1: 0, -1: 1})
        
        print(len(list(IR_df_temp[IR_df_temp['is_anomaly']==1].index)))   
        indexies_all=list(IR_df_temp[IR_df_temp['is_anomaly']==1].index)
        results=[]
        n_iterations=[]
        for train_index, test_index in cv.split(X, y):
            train_X, valid_X = X.loc[train_index].drop(indexies_all,axis=0,errors="ignore"), X.loc[test_index]
            train_y, valid_y = y.loc[train_index].drop(indexies_all,axis=0,errors="ignore"), y.loc[test_index]
           
            model = lgb.LGBMRegressor(**param_LGBM_get_score).fit(train_X,train_y,
                                                                  eval_set=[(valid_X,valid_y)],
                                                                  callbacks=[lgb.log_evaluation(period=0, show_stdv=False)])
            result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
            results.append(result)
        n=sum(results)/len(results)   
        return n
    
    
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                        direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


X_for_test=X.copy()


for n in X.columns: 
    #X[f'{n}_bins']=pd.cut(X[n],labels=[1, 2, 3,4,5],bins=5).astype('int8')
    X[f'{n}_round3']=round(X[n],3)


for n in test.columns: 
    #X[f'{n}_bins']=pd.cut(X[n],labels=[1, 2, 3,4,5],bins=5).astype('int8')
    test[f'{n}_round3']=round(test[n],3)


X['MoodEnergy'] = X['MoodScore'] * X['Energy']
X['LoudnessQuality'] = X['AudioLoudness'] * X['AcousticQuality']
epsilon = 1e-6 
X['VocalInstrumentalRatio'] = X['VocalContent'] / (X['InstrumentalScore'] + epsilon)


test['MoodEnergy'] = test['MoodScore'] * test['Energy']
test['LoudnessQuality'] = test['AudioLoudness'] * test['AcousticQuality']
epsilon = 1e-6 
test['VocalInstrumentalRatio'] = test['VocalContent'] / (test['InstrumentalScore'] + epsilon)


#####################
### Consolidation ###
#####################
models={}
preds_val={}
preds_test={}
#######################
### X for modelling ###
#######################
X_model=X.copy()
test_model = test.copy()
y_model = y.copy()


indexies=[]
for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            indexies.extend(test_index)

X_reindex=X_model.reindex(indexies,axis=0).reset_index(drop=True)
y_reindex=y_model.reindex(indexies,axis=0).reset_index(drop=True)


model_name='LGBM'
X = X_model
test = test_model
y=y_model


def objective(trial):
    param = {
     'device': trial.suggest_categorical("device",[method_LGBM]),
     "metric":trial.suggest_categorical("metric", LGBM_metrics[metric]), 
     'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
     'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
     'learning_rate': trial.suggest_float('learning_rate', 0.001,0.1),
     'num_leaves': trial.suggest_int('num_leaves', 2, 512),
     'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
     'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
     'early_stopping_round' : trial.suggest_int('early_stopping_round', 200, 200),
     'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
     'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
     'n_estimators' : trial.suggest_int('n_estimators', 1000, 10000),
     "subsample":trial.suggest_categorical("subsample", [None]),
     "subsample_freq":trial.suggest_categorical("subsample_freq", [None]),
     "reg_alpha":trial.suggest_categorical("reg_alpha", [None]),
     "colsample_bytree":trial.suggest_categorical("colsample_bytree", [None]),
     "reg_lambda":trial.suggest_categorical("reg_lambda", [None]),
     #'class_weight':weights,
     'verbosity': -1,
    }
    
    results=[]
    iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        
        model = lgb.LGBMRegressor(**param).fit(train_X,train_y,
                                            eval_set=[(valid_X,valid_y)],
                                            eval_metric=LGBM_metrics[metric],
                                            callbacks=[lgb.log_evaluation(period=0, show_stdv=False)])     
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        best_iter = model.best_iteration_
        print (best_iter)
    n=sum(results)/len(results)   
    return n

if  optuna_study == optuna_models[model_name]:
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


LGBM1_final_params = {  'device': [method_LGBM], 
                        'metric': LGBM_metrics[metric], 
                        'lambda_l1': 0.6456495189814393, 
                        'lambda_l2': 0.2801179499076938, 
                        'learning_rate': 0.084241018662707, 
                        'num_leaves': 14, 
                        'feature_fraction': 0.8696508314296023, 
                        'bagging_fraction': 0.870104623916734, 
                        'early_stopping_round': 200, 
                        'bagging_freq': 7, 
                        'min_child_samples': 54, 
                        'n_estimators': 8734, 
                        'subsample': None, 
                        'subsample_freq': None, 
                        'reg_alpha': None, 
                        'colsample_bytree': None, 
                        'reg_lambda': None,
                        'verbosity': -1}


LGBM1_final_params_full = {  'device':[method_LGBM], 
                             'metric': LGBM_metrics[metric], 
                             'lambda_l1': 0.6456495189814393, 
                             'lambda_l2': 0.2801179499076938, 
                             'learning_rate': 0.084241018662707, 
                             'num_leaves': 14, 
                             'feature_fraction': 0.8696508314296023, 
                             'bagging_fraction': 0.870104623916734, 
                             #'early_stopping_round': 200, 
                             'bagging_freq': 7, 
                             'min_child_samples': 54, 
                             'n_estimators': 65, 
                             'subsample': None, 
                             'subsample_freq': None, 
                             'reg_alpha': None, 
                             'colsample_bytree': None, 
                             'reg_lambda': None,
                             'verbosity': -1}


LGBM2_final_params= {'device': [method_LGBM], 
                     'metric': LGBM_metrics[metric], 
                     'lambda_l1': 0.6899157072650682, 
                     'lambda_l2': 7.089480514039986, 
                     'learning_rate': 0.022637385749920517, 
                     'num_leaves': 2, 
                     'feature_fraction': 0.8618728744941895, 
                     'bagging_fraction': 0.9575924933374061, 
                     'early_stopping_round': 200, 
                     'bagging_freq': 3, 
                     'min_child_samples': 59, 
                     'n_estimators': 9122, 
                     'subsample': None, 
                     'subsample_freq': None, 
                     'reg_alpha': None, 
                     'colsample_bytree': None, 
                     'reg_lambda': None,
                     'verbosity': -1}


LGBM2_final_params_full= {'device': [method_LGBM], 
                         'metric': LGBM_metrics[metric], 
                         'lambda_l1': 0.6899157072650682, 
                         'lambda_l2': 7.089480514039986, 
                         'learning_rate': 0.022637385749920517, 
                         'num_leaves': 2, 
                         'feature_fraction': 0.8618728744941895, 
                         'bagging_fraction': 0.9575924933374061, 
                         #'early_stopping_round': 200, 
                         'bagging_freq': 3, 
                         'min_child_samples': 59, 
                         'n_estimators': 746, 
                         'subsample': None, 
                         'subsample_freq': None, 
                         'reg_alpha': None, 
                         'colsample_bytree': None, 
                         'reg_lambda': None,
                         'verbosity': -1}


LGBM3_final_params={'device': [method_LGBM], 
                    'metric': LGBM_metrics[metric], 
                    'lambda_l1': 0.4671528301518596, 
                    'lambda_l2': 7.598106688500071, 
                    'learning_rate': 0.005017948537114439, 
                    'num_leaves': 32, 
                    'feature_fraction': 0.8024260896091625, 
                    'bagging_fraction': 0.9840123537766384, 
                    'early_stopping_round': 200, 
                    'bagging_freq': 5, 
                    'min_child_samples': 55, 
                    'n_estimators': 9083, 
                    'subsample': None, 
                    'subsample_freq': None, 
                    'reg_alpha': None, 
                    'colsample_bytree': None, 
                    'reg_lambda': None,
                    'verbosity': -1}


LGBM3_final_params_full={   'device': [method_LGBM], 
                            'metric': LGBM_metrics[metric], 
                            'lambda_l1': 0.4671528301518596, 
                            'lambda_l2': 7.598106688500071, 
                            'learning_rate': 0.005017948537114439, 
                            'num_leaves': 32, 
                            'feature_fraction': 0.8024260896091625, 
                            'bagging_fraction': 0.9840123537766384, 
                           #'early_stopping_round': 200, 
                            'bagging_freq': 5, 
                            'min_child_samples': 55, 
                            'n_estimators': 547, 
                            'subsample': None, 
                            'subsample_freq': None, 
                            'reg_alpha': None, 
                            'colsample_bytree': None, 
                            'reg_lambda': None,
                            'verbosity': -1}


params_best_set=[[LGBM1_final_params,LGBM1_final_params_full],
                 [LGBM2_final_params,LGBM2_final_params_full],
                 [LGBM3_final_params,LGBM3_final_params_full]]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            
            model = lgb.LGBMRegressor(**params[0]).fit(train_X,train_y,
                                                     eval_set=[(valid_X,valid_y)],
                                                     callbacks=[lgb.log_evaluation(period=0, show_stdv=False)])
            
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {sum(results)/len(results)}')
        
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
        
        model = lgb.LGBMRegressor(**params[1]).fit(X_reindex,y_reindex)
        #preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
        preds_test[f'{model_name}_{z}_{j}']=model.predict(test)

        k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)
        print(f'full result {k}')


model_name='XGB'
X = X_model
test = test_model
y=y_model


def objective(trial):
    param = {  
        'device':trial.suggest_categorical('device',[method_XGB]),
        'objective': trial.suggest_categorical('objective',[XGB_tasks[task]]),
        'eval_metric': trial.suggest_categorical('eval_metric',[XGB_metrics[metric]]),
        'lambda': trial.suggest_float('lambda', 0, 10.0),
        'alpha': trial.suggest_float('alpha', 0, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1,1.0),
        'subsample': trial.suggest_float('subsample', 0.2,1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001,0.1),
        'n_estimators': trial.suggest_int('n_estimators', 1000,10000),
        'max_depth': trial.suggest_categorical('max_depth', [2,3,4,5,6,7,8,9,10]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree',1,1),
        'early_stopping_rounds':trial.suggest_int('early_stopping_rounds',200,200),
       # 'scale_pos_weight':weight[0],
        }
    results=[]
    n_iterations=[]  
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = xgb.XGBRegressor(**param).fit(train_X,train_y,
                                      eval_set=[(valid_X,valid_y)],
                                      verbose=0)  
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        best_iter = model.best_iteration
        print (best_iter)
    n=sum(results)/len(results) 
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=10000)


XGB1_final_params={  'device': method_XGB, 
                     'objective': XGB_tasks[task], 
                     'eval_metric': XGB_metrics[metric], 
                     'lambda': 4.319912919451416, 
                     'alpha': 3.53166054842918, 
                     'colsample_bytree': 0.24826143747765927, 
                     'subsample': 0.7854254811493886, 
                     'learning_rate': 0.017410659228746746, 
                     'n_estimators': 2181, 
                     'max_depth': 6, 
                     'min_child_weight': 2, 
                     'num_parallel_tree': 1, 
                     'early_stopping_rounds': 200,
                     'verbose' : 0
                    }


XGB1_final_params_full={ 'device': method_XGB, 
                         'objective': XGB_tasks[task], 
                         'eval_metric': XGB_metrics[metric], 
                         'lambda': 4.319912919451416, 
                         'alpha': 3.53166054842918, 
                         'colsample_bytree': 0.24826143747765927, 
                         'subsample': 0.7854254811493886, 
                         'learning_rate': 0.017410659228746746, 
                         'n_estimators': 220, 
                         'max_depth': 6, 
                         'min_child_weight': 2, 
                         'num_parallel_tree': 1, 
                        # 'early_stopping_rounds': 200,
                                         'verbose' : 0
                        }


XGB2_final_params= {'device': method_XGB, 
                    'objective': XGB_tasks[task], 
                    'eval_metric': XGB_metrics[metric], 
                    'lambda': 6.9217464610279515, 
                    'alpha': 0.023096512814280594, 
                    'colsample_bytree': 0.5213223100837274, 
                    'subsample': 0.7868498962835917, 
                    'learning_rate': 0.025718737547928347, 
                    'n_estimators': 2097, 
                    'max_depth': 2, 
                    'min_child_weight': 1, 
                    'num_parallel_tree': 1, 
                    'early_stopping_rounds': 200,
                                     'verbose' : 0
                   }


XGB2_final_params_full= {   'device': method_XGB, 
                            'objective': XGB_tasks[task], 
                            'eval_metric': XGB_metrics[metric], 
                            'lambda': 6.9217464610279515, 
                            'alpha': 0.023096512814280594, 
                            'colsample_bytree': 0.5213223100837274, 
                            'subsample': 0.7868498962835917, 
                            'learning_rate': 0.025718737547928347, 
                            'n_estimators': 615, 
                            'max_depth': 2, 
                            'min_child_weight': 1, 
                            'num_parallel_tree': 1, 
                            #'early_stopping_rounds': 200,
                                          'verbose' : 0
                           }


XGB3_final_params={ 'device': method_XGB, 
                    'objective': XGB_tasks[task], 
                    'eval_metric': XGB_metrics[metric], 
                    'lambda': 5.697983758917905, 
                    'alpha': 2.0991284692298984, 
                    'colsample_bytree': 0.7054170544599856, 
                    'subsample': 0.8532430980725452, 
                    'learning_rate': 0.06754138107693969, 
                    'n_estimators': 1518, 
                    'max_depth': 5, 
                    'min_child_weight': 4, 
                    'num_parallel_tree': 1, 
                    'early_stopping_rounds': 200,
                                    'verbose' : 0
                   }


XGB3_final_params_full={'device': method_XGB, 
                        'objective': XGB_tasks[task], 
                        'eval_metric': XGB_metrics[metric], 
                        'lambda': 5.697983758917905, 
                        'alpha': 2.0991284692298984, 
                        'colsample_bytree': 0.7054170544599856, 
                        'subsample': 0.8532430980725452, 
                        'learning_rate': 0.06754138107693969, 
                        'n_estimators': 56, 
                        'max_depth': 5, 
                        'min_child_weight': 4, 
                        'num_parallel_tree': 1, 
                        #'early_stopping_rounds': 200,
                         'verbose' : 0
                       }


params_best_set=[[XGB1_final_params,XGB1_final_params_full],
                 [XGB2_final_params,XGB2_final_params_full],
                 [XGB3_final_params,XGB3_final_params_full]]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            print(test_index)
            model = xgb.XGBRegressor(**params[0]).fit(train_X,train_y,
                                              eval_set=[(train_X,train_y),(valid_X,valid_y)],
                                              verbose=200)  
            
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {sum(results)/len(results)}')
        
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
        
        model = xgb.XGBRegressor(**params[1]).fit(X_reindex,y_reindex)
       # preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
        preds_test[f'{model_name}_{z}_{j}']=model.predict(test)

        k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)
        print(f'full result {k}')


model_name='CAT'
X = X_model
test = test_model
y=y_model


def objective(trial):
    param = {
        'task_type':trial.suggest_categorical("task_type",[method_CAT]),
        "loss_function":trial.suggest_categorical("loss_function",[CAT_tasks[task]]),
        'eval_metric':trial.suggest_categorical("eval_metric",[CAT_metrics[metric]]),
        #"rsm":trial.suggest_float("rsm", 0.5, 1),
        'use_best_model':trial.suggest_categorical("use_best_model", [True]) ,
        "iterations":trial.suggest_int("iterations", 1000, 10000),
        "learning_rate":trial.suggest_float("learning_rate", 0.001,0.1),
        "depth":trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        "bootstrap_type":trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        "random_strength":trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature":trial.suggest_float("bagging_temperature", 0, 10.0),
        "od_type":trial.suggest_categorical("od_type", ["Iter"]),
        "od_wait":trial.suggest_int("od_wait", 200, 200),
        #'scale_pos_weight':weight[1],
        "verbose":0}
    results=[]
    n_iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = CatBoostRegressor(**param).fit(train_X,train_y,
                                            eval_set=[(valid_X,valid_y)],
                                            verbose=0)  
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        best_iter = model.get_best_iteration()
        print (best_iter)
    n=sum(results)/len(results)    
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


CAT1_final_params= {'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric], 
                    'use_best_model': True, 
                    'iterations': 7683, 
                    'learning_rate': 0.09658122307277964, 
                    'depth': 4, 
                    'l2_leaf_reg': 2.288793143798914, 
                    'bootstrap_type': 'Bayesian', 
                    'random_strength': 0.0005022781372817699, 
                    'bagging_temperature': 0.3120792954189741, 
                    'od_type': 'Iter', 
                    'od_wait': 200,
                    'verbose': False
                   }


CAT1_final_params_full = { 'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric],  
                            #'use_best_model': True, 
                            'iterations': 131, 
                            'learning_rate': 0.09658122307277964, 
                            'depth': 4, 
                            'l2_leaf_reg': 2.288793143798914, 
                            'bootstrap_type': 'Bayesian', 
                            'random_strength': 0.0005022781372817699, 
                            'bagging_temperature': 0.3120792954189741, 
                            'od_type': 'Iter',
                            'verbose':False
                           # 'od_wait': 200
                           }


CAT2_final_params = {'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric], 
                     'use_best_model': True, 
                     'iterations': 7486, 
                     'learning_rate': 0.0050762441391279944, 
                     'depth': 4, 
                     'l2_leaf_reg': 4.0820515869478475, 
                     'bootstrap_type': 'Bayesian', 
                     'random_strength': 0.03873977347382524, 
                     'bagging_temperature': 0.1950575020300821, 
                     'od_type': 'Iter', 
                     'od_wait': 200,
                     'verbose':False
                    }


CAT2_final_params_full = {'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric], 
                     #'use_best_model': True, 
                     'iterations': 1460, 
                     'learning_rate': 0.0050762441391279944, 
                     'depth': 4, 
                     'l2_leaf_reg': 4.0820515869478475, 
                     'bootstrap_type': 'Bayesian', 
                     'random_strength': 0.03873977347382524, 
                     'bagging_temperature': 0.1950575020300821, 
                     'od_type': 'Iter', 
                     'verbose':False
                     #'od_wait': 200
                    }


CAT3_final_params = {'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric], 
                     'use_best_model': True, 
                     'iterations': 8892, 
                     'learning_rate': 0.054203183404353455, 
                     'depth': 4, 
                     'l2_leaf_reg': 0.873799695519857, 
                     'bootstrap_type': 'Bayesian', 
                     'random_strength': 1.6196698744001308e-05, 
                     'bagging_temperature': 0.5870940815448663, 
                     'od_type': 'Iter', 
                     'od_wait': 200,
                     'verbose':False
                    }


CAT3_final_params_full = { 'task_type': method_CAT, 
                    'loss_function': CAT_tasks[task], 
                    'eval_metric': CAT_metrics[metric], 
                           #  'use_best_model': True, 
                             'iterations': 247, 
                             'learning_rate': 0.054203183404353455, 
                             'depth': 4, 
                             'l2_leaf_reg': 0.873799695519857, 
                             'bootstrap_type': 'Bayesian', 
                             'random_strength': 1.6196698744001308e-05, 
                             'bagging_temperature': 0.5870940815448663, 
                             'od_type': 'Iter', 
                             'verbose':False
                             #'od_wait': 200
                            }


params_best_set=[[CAT1_final_params,CAT1_final_params_full],
                 [CAT2_final_params,CAT2_final_params_full],
                 [CAT3_final_params,CAT3_final_params_full]]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]

            model = CatBoostRegressor(**params[0]).fit(train_X,train_y,
                                                eval_set=[(valid_X,valid_y)],
                                                verbose=False)  
                    
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {sum(results)/len(results)}')
        
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
        
        model = CatBoostRegressor(**params[1]).fit(X_reindex,y_reindex)
        
       # preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
        preds_test[f'{model_name}_{z}_{j}']=model.predict(test)

        k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)
        print(f'full result {k}')


model_name='AB'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        'n_estimators':trial.suggest_int("n_estimators", 100, 2000,step=100),
        "learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)}
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = AdaBoostClassifier(**param).fit(train_X,train_y)  
        result = roc_auc_score(valid_y,(model.predict(valid_X)))
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best = {}


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = AdaBoostClassifier(**params_best).fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final AB Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='GB'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        #"loss":trial.suggest_categorical("loss", [loss_object]),
        "learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'max_depth':trial.suggest_int("max_depth", 1, 9,step=1),
        'n_estimators':trial.suggest_int("n_estimators", 5, 500,step=5),
        'verbose':0
    }
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = GradientBoostingClassifier(**param).fit(train_X,train_y)  
        result = roc_auc_score(valid_y,(model.predict(valid_X)))
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best = {}


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = GradientBoostingClassifier(**params_best).fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final GB Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='ET'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        'n_estimators':trial.suggest_int("n_estimators", 10, 250,step=5),
        'criterion':trial.suggest_categorical("criterion", ['gini','entropy']), 
        'max_depth':trial.suggest_int("max_depth", 3, 8,step=1),        
        'min_samples_split':trial.suggest_int("min_samples_split", 2, 4,step=1),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 1, 4,step=1),        
        'n_estimators':trial.suggest_int("n_estimators", 10, 250,step=5),
        'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
        'class_weight':trial.suggest_categorical("class_weight", [weights]),
        'verbose':trial.suggest_categorical("verbose", [0]),
        'max_features':trial.suggest_categorical("max_features", ['sqrt','log2']),
        'verbose':0}
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = ExtraTreesClassifier(**param).fit(train_X,train_y)  
        result = roc_auc_score(valid_y,(model.predict(valid_X)))
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best = {}


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = ExtraTreesRegressor(**params_best).fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final ET Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='RF'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        'criterion':trial.suggest_categorical("criterion", [loss]), 
        'n_estimators':trial.suggest_int("n_estimators", 10, 300,step=5),
        'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
        #'class_weight':trial.suggest_categorical("class_weight", [weights]),
        'verbose':trial.suggest_categorical("verbose", [0]),
        'max_depth':trial.suggest_int("max_depth", 3, 12,step=1),
        'min_samples_split':trial.suggest_int("min_samples_split", 2, 4,step=1),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 1, 4,step=1),
       # 'max_features':trial.suggest_categorical("max_features", ['sqrt','log2']),
        'verbose':0}
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestClassifier(**param).fit(train_X,train_y)  
        result = roc_auc_score(valid_y,model.predict_proba(valid_X)[:,1])
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best={#'criterion':objective,
             'n_jobs': -1, 
             'verbose': 0,
             'class_weight':weights}


params_best= {'criterion': 'entropy', 
              'n_estimators': 300, 
              'n_jobs': -1, 
              'verbose': 0, 
              'max_depth': 20, #'max_depth': 12
              'min_samples_split': 3, 
              'min_samples_leaf': 2}


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestClassifier(**params_best).fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=((model.predict_proba(valid_X)[:,1]))
        preds_test[f'{model_name}_{i}_{j}']=((model.predict_proba(test)[:,1]))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final RF Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='LCV'
X = X_model
test = test_model
y=y_model


def objective(trial):
    print("START__________________________________")       
    param = {
        'precompute':"auto",
        'fit_intercept':True,
        'normalize':False,
        'max_iter':1000,
        'verbose':False,
        'eps':1e-04,
        'cv':cv,
        'n_alphas':1000,
        'n_jobs':8
        }
    results=[]
    n_iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = LassoCV(**params).fit(train_X,train_y,
                                            eval_set=[(valid_X,valid_y)],
                                            verbose=0
                                             )  
        result = roc_auc_score(valid_y,((model.predict_proba(valid_X)[:,1])))
        results.append(result)
    print("Average n_ite=" + str(i))
    n=sum(results)/len(results)
    print(n)
    print("FIIINISH__________________________________\n")    
            
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


 params_best = {}


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
       
        model = linear_model.LassoCV(*params_best).fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final LCV Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='LR'
X = X_model
test = test_model
y = y_model


if gpu_switch == "ON":
    method = "GPU"
else:
    method = "CPU"

def objective(trial):
    print("START__________________________________")       
    param = {
       # 'precompute':"auto",
       # 'fit_intercept':True,
       # 'normalize':False,
        #'max_iter':1000,
        #'verbose':False,
        #'eps':1e-04,
       # 'cv':cv,
        #'n_alphas':1000,
        #'task_type':method,
       # "#loss_function":trial.suggest_categorical("loss_function", ["Logloss"]),
        
       # 'penalty':trial.suggest_categorical("penalty", ['l1', 'l2', 'elasticnet', 'none']) ,
       # 'solver':trial.suggest_categorical("solver", ['lbfgs','newton-cg','liblinear','sag','saga']) ,
        "max_iter":trial.suggest_int("max_iter", 100, 4000), 
        #"C":trial.suggest_float("C", 1e-4, 4, log=True),
        'class_weight':{0:1,1:1}
    }
        
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = LogisticRegression(**param).fit(train_X,train_y)
        result = roc_auc_score(valid_y,((model.predict_proba(valid_X)[:,1])))
        results.append(result)
    n=sum(results)/len(results)       
    return n

if  optuna_study==optuna_models['LR']:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best = {"max_iter":4000, 
               'class_weight':weights
              }


if finish_models[model_name] and finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
       
        model = LogisticRegression(**params_best).fit(train_X,train_y)
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final LR Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='KNC'
X = X_model
test = test_model
y = y_model


def objective(trial):
    param = {
    'n_neighbors' : trial.suggest_int("n_neighbors", 1, 500),
    'leaf_size' : trial.suggest_int("leaf_size", 1, 500),
    'weights' : trial.suggest_categorical("weights", ['uniform', 'distance']),
    'p' : trial.suggest_categorical("p", [1, 2, 3, 4, 5]),
    'metric' : trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])}
 
    results=[]
    n_iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        
        #train_X,train_y = sm.fit_resample(train_X,train_y)
        
        model = KNeighborsClassifier(**param).fit(train_X,train_y)  
        result = metric_call()
        results.append(result)
    n=sum(results)/len(results)
    print(n)
    print("FIIINISH__________________________________\n")    
            
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_KNC_best={'n_neighbors': 77, 
                 'leaf_size': 343, 
                 'weights': 'distance', 
                 'p': 3, 
                 'metric': 'manhattan'}


if finish_models[model_name] == finish_set:
    results=[]
    n_iterations=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        j=i//n_splits
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        
        model = KNeighborsRegressor().fit(train_X,train_y)  
        
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final KNC Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='SVC'
X = X_model
test = test_model
y = y_model


def objective(trial):
    param= {
            'kernel': trial.suggest_categorical("kernel", ['linear', 'rbf', 'poly']), 
            'gamma': trial.suggest_categorical("gamma", [0.001, 0.01, 0.1, 1, 10, 100]),
            'C': trial.suggest_categorical("C", [0.001, 0.01, 0.1, 1,10,100,1000]),
            'degree': trial.suggest_categorical("degree", [0, 1, 2, 3, 4, 5, 6]),
            'probability':trial.suggest_categorical("probability", [True]),
    }
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
              
        model = SVC(**param).fit(train_X,train_y)  
        result = metric_call()
        results.append(result)
    n=sum(results)/len(results)
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


params_best= {'kernel': 'poly', 
              'degree': 3, 
              'C':100, 
              'epsilon': 0.01
                   }


if finish_models[model_name] == finish_set:
    results=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        start_time = time.time()
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = SVR(**params_best).fit(train_X,train_y)  
                
        models[f'{model_name}_{i}_{j}']=(model)
        preds_val[f'{model_name}_{i}_{j}']=(model.predict(valid_X))
        preds_test[f'{model_name}_{i}_{j}']=(model.predict(test))
        
        result = metric_call()
        results.append(result)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
    print (f'\033[0;35;40m Final LGB Result = {sum(results)/len(results)} \033[0;30;0m')


model_name='KERAS'
X = X_model
test = test_model
y = y_model


X =X.to_numpy()
test = test.to_numpy()
y = y_model.to_numpy()
input_shape = [X.shape[1]]


opt = tf.optimizers.Adam(learning_rate=0.001)
reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', 
                                                      factor=0.8, 
                                                      patience=3, 
                                                      verbose=0,
                                                      min_delta=0.0001 ,
                                                      min_lr=0.000001, 
                                                      mode='max')

early_stopping = keras.callbacks.EarlyStopping(patience=8,
                                               verbose=0,
                                               min_delta=0.0001,
                                               monitor="val_auc",
                                               mode='max',
                                               restore_best_weights=True)

def get_model(trial):
    opt = tf.optimizers.Adam(learning_rate=0.001)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', 
                                                          factor=0.8, 
                                                          patience=3, 
                                                          verbose=0,
                                                          min_delta=0.0001 ,
                                                          min_lr=0.000001, 
                                                          mode='max')

    early_stopping = keras.callbacks.EarlyStopping(patience=8,
                                                   verbose=0,
                                                   min_delta=0.0001,
                                                   monitor="val_auc",
                                                   mode='max',
                                                   restore_best_weights=True)
    model = keras.Sequential()
    activation=trial.suggest_categorical("activation_type_is_", ['relu','selu'])
    model.add(layers.BatchNormalization(input_shape = [X.shape[1]]))
    
    for i in np.arange(0,trial.suggest_int("num_layers", 1, 6)):
        model.add(layers.Dense(units = trial.suggest_int("units_"+str(i+1), 32, 512, step =32),
                               activation=activation))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(trial.suggest_float("drop_"+str(i+1), 0.05, 0.25,step=0.05))) 
   
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
    optimizer=opt,
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.80, gamma=2.0),
    #loss='binary_crossentropy',
    metrics=['AUC'])  
    weights_before_update=model.get_weights()
    
    preds=[]
    results=[]
    i=0
    for train_index, test_index in (cv.split(X, y)):
        opt = tf.optimizers.Adam(learning_rate=0.001)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', 
                                                              factor=0.8, 
                                                              patience=5, 
                                                              verbose=0,
                                                              min_delta=0.0001 ,
                                                              min_lr=0.000001, mode='max')

        early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                                       min_delta=0.0001,
                                                       monitor="val_auc",
                                                       mode='max',
                                                       restore_best_weights=True)
    
        i+=1
        start_time=time.time()
        train_X, valid_X = X[train_index], X[test_index]
        train_y, valid_y = y[train_index], y[test_index]
        history = model.fit(
            train_X,train_y,
            validation_data=(valid_X,valid_y),
            batch_size=512*20,
            epochs=200,
            class_weight = weights,
        callbacks=[early_stopping,
                   reduce_lr_loss],
        verbose=0)
        
        pred = (model.predict(valid_X))
        preds.append(pred)
        result = roc_auc_score(valid_y,pred)
        
        results.append(result)
        model.set_weights(weights_before_update)
        n=sum(results)/len(results)
        from keras import backend as K 
        K.clear_session()
    return n


if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(get_model, n_trials=1000)
    print('Best trial:', study.best_trial.params)


opt = tf.optimizers.Adam(learning_rate=0.001)


def get_model():
    model = keras.Sequential([
            layers.BatchNormalization(input_shape = [X.shape[1]]),
            layers.Dense(288,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),        
            layers.Dense(320,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(320,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.05),
            layers.Dense(32,activation='selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.05),  
            layers.Dense(1,activation='sigmoid'),
    ])
    model.compile(
    optimizer=opt,
       # tf.keras.losses.mean_squared_error
    #loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.1, gamma=3),
    loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    #metrics=['AUC']
    )  
    
    return model


if finish_models[model_name] == finish_set:
    preds=[]
    results=[]
    predicts=pd.DataFrame()
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        opt = tf.optimizers.Adam(learning_rate=0.001)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', 
                                                              factor=0.8, 
                                                              patience=5, 
                                                              verbose=0,
                                                              min_delta=0.0001 ,
                                                              min_lr=0.000001, mode='max')

        early_stopping = keras.callbacks.EarlyStopping(patience=30,
                                                       min_delta=0.0001,
                                                       monitor="val_auc",
                                                       mode='max',
                                                       restore_best_weights=True)

        start_time = time.time()
        train_X, valid_X = X[train_index], X[test_index]
        train_y, valid_y = y[train_index], y[test_index]
        model = get_model()
        history = model.fit(
            train_X,train_y,
            validation_data=(valid_X,valid_y),
            batch_size=512*10,
            epochs=400,
            class_weight = weights,
            callbacks=[early_stopping,reduce_lr_loss],
        verbose=1)
        pred = (model.predict(valid_X))
        pred_test = (model.predict(test))
        preds.append(pred)
        predicts[i]=pd.DataFrame(pred_test)
        preds_val[f'{model_name}_{i}_{j}']=model.predict(valid_X).reshape(1,-1)[0]
        preds_test[f'{model_name}_{i}_{j}']=model.predict(test).reshape(1,-1)[0]
        result = round(roc_auc_score(valid_y,pred),4)
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
        results.append(result)
    print (f'\033[0;35;40m Final KERAS result = {(sum(results)/len(results))} \033[0;30;0m')


pip install autokeras


import autokeras as ak


X = X_model
test = test_model
model_name='AUTOKERAS'


if finish_models[model_name] and finish_set:
    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)
    model = ak.ImageClassifier()
    model.fit(X,y)
    mean_squared_error(model.predict(X),y)**(1/2)


pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o


import h2o
h2o.init()


from h2o.automl import H2OAutoML


model_name='H2O'
X = X_model
test = test_model


if finish_models[model_name] and finish_set:
    h2o_train=X.copy()
    h2o_train[target]=y
    
    h2o_train = h2o.H2OFrame(h2o_train)
    h2o_test = h2o.H2OFrame(test)
    h2o_sample = h2o.H2OFrame(sample)
    
    model = H2OAutoML(max_runtime_secs=60*60*5, seed = 3)
    model.train(x = list(X.columns), y = target, training_frame = h2o_train)
    
    output_path = "/kaggle/working/H2Oensemble.csv" 
    h2o_sample[target]=model.leader.predict(h2o_test)
    h2o.export_file(h2o_sample, path=output_path, force=True)


X = X_reindex
test = test_model
y=y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


################
### Settings ###
################
model_name='FLAML'

X = X_reindex
test = test_model

y=y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


pip install flaml


pip install flaml[spark]


from flaml import AutoML as FlamlAutoML


if finish_models[model_name] and finish_set:
    model = FlamlAutoML()
    model.fit(X_train=X_ensemble, 
               y_train = y, 
               task="regression",
               n_jobs=4,
               metric='rmse',
               time_budget=60*60*9,
               verbose=0,
               ensemble=True,
               eval_method='cv',
               n_splits =7,
               use_spark  = False
              )
    print(mean_squared_error(model.predict(X_ensemble),y)**(1/2))
    sample[target]=model.predict(test_ensemble)
    sample.to_csv('FLAML_Ensemble.csv', index=False)


print('finish')

