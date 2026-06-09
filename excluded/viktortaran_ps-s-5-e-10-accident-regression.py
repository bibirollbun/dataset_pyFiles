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
#import cudf.pandas
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


test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test.drop("id",axis=1,inplace=True)
test['adv_val'] =  0

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train.drop("id",axis=1,inplace=True)
train['adv_val'] =  1

original = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
original['adv_val'] =  1

sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


sets={'train':train,'test':test,'original':original}


temp=pd.DataFrame()
for q in range(8):
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

gpu_switch = True

if gpu_switch:
    method_LGBM = "gpu"
    method_XGB = "cuda"
    method_CAT = "GPU"
else:
    method_LGBM = "cpu"
    method_XGB = "cpu" 
    method_CAT = "CPU"

plt.style.use('dark_background')
k_fold_result_round=5
##########
### CV ###
##########

n_splits = 6
n_repeats =1
#cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats = n_repeats, random_state=2023)
#cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#cv = GroupKFold(n_splits=n_splits)
cv = RepeatedKFold(n_splits=n_splits, n_repeats = n_repeats, random_state=2024)

########################
### Define Weights   ###
########################

#weights = {0: 0.5009553158705701, 1: 262.19354838709677}

#4 DROPPING DUPLICATES
drop_dup=True

#4.1 ENCODING AND IMPUTING
target='accident_risk'

#4.2 VISUALIZATION
vis_hist_active=False
vis_boxplot_active=False
vis_scatter_active=False
vis_scatter_exp_active=False

#4.3 ADVERSARIAL VALIDATION
cv_detection_switch=False

#4.3 ADVERSARIAL VALIDATION
adv_val=False
adv_val_perm_active=False
concat_with_orig=False

#4.5 PERMUTATION_IMPORTANCE
perm_switch=False
own_technique_switch=False

#4.6 NORMALIZATION AND SCALING
normalization=False

#4.7 OUTLIERS

#4.7.1 IQR
IQR_tuning_switch=False
IQR_apply_switch=False

#4.7.2 IF
IR_tuning_switch=False



#5 MODELING #6 ENSEMBLING

#######################
### Optuna Settings ###
#######################
direction = "minimize"
optuna_study = True 
optuna_models={'LGBM' : False, 
               'XGB'  : False,
               'CAT'  : False,
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
               'ET'   : True,
               'RF'   : True,
               'LCV'  : False,
               'LR'   : False,
               'KNC'  : False,
               'SVC'  : False,
               'KERAS': False,
               'AUTOKERAS'  : False,
               'H2O'  : False}
#######################
### Ensemble Settings ###
#######################
ensemble_set = True
ensemble_models={'FLAML' : False,
                 'WE_Ensemble': True,
                 'RidgeCV_Ensemble':False,
                 'LR_Ensemble':False,
                 'YDF':False,
                 'KERAS':False,
                 'XGB_Ensemble':False
                 }

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

ET_metrics=  {'Root Mean Squared Error':'squared_error'}

RF_metrics=  {'Root Mean Squared Error':'squared_error'}

KERAS_metrics=  {'Root Mean Squared Error':'mean_squared_error'}

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
        print (f'\033[0;33;40m Step#{i} Result = {result} \033[0;30;0m')
    print (f'\033[0;35;40m Final LGBM Result = {sum(results)/len(results)} \033[0;30;0m')
    print (f'time is {np.round((time.time()-start_time),1)}')


###############################################
### The function for step-by-step analyzing ###
###############################################

param_XGB_get_score = { 'device':method_XGB,
                        'eval_metric': XGB_metrics[metric],
                        'learning_rate': 0.05,
                        'n_estimators': 20000,
                        'early_stopping_rounds':200,
                        'enable_categorical':True}
        
def XGB_get_score(X,y):
    results=[]
    iters=[]
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = xgb.XGBRegressor(**param_XGB_get_score).fit(train_X,train_y,
                                                  eval_set=[(valid_X,valid_y)],
                                                  verbose=0)  
        result=round(mean_squared_error(valid_y,model.predict(valid_X))**(1/2),6)
        results.append(result)
        best_iter = model.best_iteration
        iters.append(best_iter)
        print (f'\033[0;33;40m Step#{i} Result = {result} with {best_iter} iterations \033[0;30;0m')
    print (f'\033[0;35;40m XGB Result = {round(sum(results)/len(results),6)} with {sum(iters)/len(iters)} iterations  \033[0;30;0m')



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
        print (f'\033[0;33;40m Step#{i} Result = {result} \033[0;30;0m')
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


cat_cols = ['road_type','lighting','weather','road_signs_present',
            'public_road','time_of_day','holiday','school_season']            
drop_col = []
remain_col =['accident_risk']
num_cols=['curvature','num_lanes','speed_limit','num_reported_accidents']


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
train[cat_cols] = pd.DataFrame(cat_imp.fit_transform(train[cat_cols]),columns=cat_cols)
temp=pd.DataFrame(ohe.fit_transform(train[cat_cols]),columns=ohe.get_feature_names_out())
train=pd.concat([train.drop(cat_cols,axis=1),temp],axis=1) 

original[num_cols] = pd.DataFrame(num_imp.transform(original[num_cols]),columns=num_cols)
original[cat_cols] = pd.DataFrame(cat_imp.transform(original[cat_cols]),columns=cat_cols)
temp=pd.DataFrame(ohe.transform(original[cat_cols]),columns=ohe.get_feature_names_out())
original=pd.concat([original.drop(cat_cols,axis=1),temp],axis=1)    

test[num_cols] = pd.DataFrame(num_imp.transform(test[num_cols]),columns=num_cols)
test[cat_cols] = pd.DataFrame(cat_imp.transform(test[cat_cols]),columns=cat_cols)
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


sns.heatmap(train.corr(),annot=False,fmt=".2f")


sns.heatmap(test.corr(),annot=False,fmt=".2f")


if vis_hist_active: #general_settings 4.2
    viz_comp(X,test,"X vs TEST")
    viz_comp(X_orig,test,"X_orig vs TEST")


if vis_boxplot_active: #general_settings 4.2
    viz_boxplot(X)
    viz_boxplot(X_orig)
    viz_boxplot(test)


if vis_scatter_active:
    fig, ax = plt.subplots(5,5, figsize=(20, 20))
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


if cv_detection_switch:
    X_temp=X.copy()
    test_temp=test.copy()
    
    y_X_temp=X_temp['adv_val']
    y_test_temp=test_temp['adv_val']
    
    X_temp = X_temp.drop(['adv_val'],axis=1)
    test_temp=test_temp.drop(['adv_val'],axis=1)
    start_time = time.time()
    for cv_search in range(2,30):
        results=[]
        print(f'#cv_search={cv_search}')
        cv2 = RepeatedKFold(n_splits=cv_search, n_repeats = 1, random_state=2024)
        cv3 = RepeatedKFold(n_splits=2, n_repeats = 1, random_state=2024)
        for i,(lv1_train, lv1_val) in enumerate(cv2.split(X)):
            X_1 = X.iloc[lv1_val]
            test
            meta=pd.concat([X_1,test],axis=0)
        
            X_2=meta.drop(['adv_val'],axis=1)
            y_2=meta['adv_val']
        
            X_2,y_2=shuffle(X_2,y_2,random_state=2024)
        
            for i,(lv2_train, lv2_val) in enumerate(cv3.split(X_2)):
                X_2_train,X_2_val  = X_2.iloc[lv2_train],X_2.iloc[lv2_val]
                y_2_train,y_2_val  = y_2.iloc[lv2_train],y_2.iloc[lv2_val]
         
                model = xgb.XGBClassifier(**param_XGB_get_score).fit(X_2_train,y_2_train,
                                                      eval_set=[(X_2_val,y_2_val)],
                                                      verbose=0)  
                result=abs(0.5-roc_auc_score(y_2_val,model.predict_proba(X_2_val)[:, 1]))
            
                results.append(result)
                
               # print (f'\033[0;33;40m Step#{i} Result = {result} \033[0;30;0m')
        print (f'\033[0;35;40m Final XGB Result = {round(sum(results)/len(results),6)} \033[0;30;0m')
        #print (f'time is {np.round((time.time()-start_time),1)}')


def adv_validation (data_1,data_2):
    X_temp=shuffle(pd.concat([data_1,data_2],ignore_index=True))  
    X_full=X_temp.drop(['adv_val'],axis=1)
    y_full=X_temp['adv_val']
    X_full,y_full=shuffle(X_full,y_full)
    X_full=X_full.reset_index(drop=True)
    y_full=y_full.reset_index(drop=True)
    result = cross_val_score(xgb.XGBClassifier(verbosity=2,
                                               device=method_XGB),X_full,y_full,scoring='roc_auc',cv=5).mean()
    return result
 


if adv_val: #general_settings 4.3
    sets_adv_val={'train':X,'original':X_orig}
    for one in sets_adv_val:
        print (f'\033[0;33;40m Result between {one} and test = {adv_validation(sets_adv_val[one],test)} \033[0;30;0m')


if adv_val_perm_active: #general_settings 4.3
    model = xgb.XGBClassifier(verbosity=1,
                             device=method_XGB
                             )
    adv_val_metric='roc_auc'
    X_orig_test_perm=pd.concat([X_orig,test]).drop(['adv_val'],axis=1).reset_index(drop=True)
    y_orig_test_perm=pd.concat([X_orig,test])['adv_val'].reset_index(drop=True)
    permute = PermutationImportance(model,random_state=2024,n_iter =1,cv=5,scoring = adv_val_metric).fit(X_orig_test_perm, y_orig_test_perm)
    eli5.show_weights(permute, feature_names = X_orig_test_perm.columns.tolist(),top=50)


X = X.drop(['adv_val'],axis=1)
X_orig = X_orig.drop(['adv_val'],axis=1)
test = test.drop(['adv_val'],axis=1)


if concat_with_orig: #general_settings 4.3
    X=pd.concat([X,X_orig],axis=0).reset_index(drop=True)
    y=pd.concat([y,y_orig],axis=0).reset_index(drop=True)


res_model_orig=LinearRegression().fit(X_orig,y_orig)


plt.figure(figsize=(10,10))
plt.scatter(y_orig,res_model_orig.predict(X_orig),s=2,color='cyan')
plt.plot([0, 1],color='red',linewidth=4)
plt.xticks(np.arange(0,1.05,0.05))
plt.yticks(np.arange(0,1.05,0.05))
plt.show


res_model_1=LinearRegression().fit(X,y)


plt.figure(figsize=(10,10))
plt.scatter(y,res_model_1.predict(X),s=2,color='cyan')
plt.plot([0, 1],color='red',linewidth=4)
plt.xticks(np.arange(0,1.05,0.05))
plt.yticks(np.arange(0,1.05,0.05))
plt.show


X=X.drop(list(y[y==1].index),axis=0)
y=y.drop(list(y[y==1].index),axis=0)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


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
    model = xgb.XGBRegressor( device=method_XGB,
                              eval_metric=XGB_metrics[metric],
                              verbosity=1,
                              learning_rate = 0.3,
                              n_estimators = 2000)
    #X=shuffle(pd.concat([original,test],ignore_index=True).drop(['adv_val',target],axis=1),random_state=2023).reset_index(drop=True)
    #y=shuffle(pd.concat([original,test],ignore_index=True).adv_val,random_state=2023).reset_index(drop=True)
    permute = PermutationImportance(model,random_state=2024,n_iter =1,cv=5,scoring = Perm_metrics[metric]).fit(X, y)
    eli5.show_weights(permute, feature_names = X.columns.tolist(),top=100)


if own_technique_switch:
    for i,n in enumerate(list(X.columns)):
        print (i)
        XGB_get_score(X[list(X.columns[:24-i])],y)


my_own_technique_list=['num_lanes',
 'curvature',
 'speed_limit',
 'num_reported_accidents',
 'road_type_rural',
 'lighting_daylight',
 'lighting_dim',
 'lighting_night',
 'weather_clear',
 'weather_foggy',
 'weather_rainy',
 'road_signs_present_False',
 'public_road_False',
 'time_of_day_evening',
 'time_of_day_morning',
 'holiday_False',
 'school_season_False']


X=X[my_own_technique_list].copy()
test=test[my_own_technique_list].copy()


def Box_transform(X,test):
    box_cols = num_cols
    for column in box_cols: 
        X_temp,fitted_lambda = stats.boxcox(X[column]) 
        X[column]=X_temp 
        test_temp = stats.boxcox(test[column],fitted_lambda) 
        test[column]=test_temp


#X[num_cols]=X[num_cols]+0.0000000000001
#test[num_cols]=test[num_cols]+0.0000000000001
#Box_transform(X,test)


def StaSca_transform(X,test):
    StaSca = StandardScaler()
    X[num_cols] = pd.DataFrame(data = StaSca.fit_transform(X[num_cols]),columns = X[num_cols].columns)
    test[num_cols] = pd.DataFrame(data = StaSca.transform(test[num_cols]),columns = test[num_cols].columns)


IQR_cols=['curvature']


if IQR_tuning_switch:
    def objective(trial):
        IQR_k_features={}
        for i,n in enumerate([IQR_cols]):
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
        print (f'\033[0;33;40m Result = {round(n,7)} \033[0;30;0m')
        return n
    #print (f'\033[0;33;40m Result = {round(n,7)} \033[0;30;0m')
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                        direction=direction)
    study.optimize(objective, n_trials=10000)
   
    print('Best trial:', study.best_trial.params)


IQR_k_params={}


if IQR_apply_switch:
    IQR_k_features={}
    for i,n in enumerate(num_cols):
        IQR_k_features[n]=[list(IQR_k_params.values())[i*2],list(IQR_k_params.values())[i*2+1]]
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


X['new']= X['curvature']*0.30 + \
(X['speed_limit']>=60)*0.20+\
(X['num_reported_accidents']>2)*0.1+\
(X['weather_foggy'])*0.1 +\
(X['weather_rainy'])*0.1 +\
(X['lighting_night'])*0.2

test['new']= test['curvature']*0.30 + \
(test['speed_limit']>=60)*0.20+\
(test['num_reported_accidents']>2)*0.1+\
(test['weather_foggy'])*0.1 +\
(test['weather_rainy'])*0.1 +\
(test['lighting_night'])*0.2



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
     'n_estimators' : trial.suggest_int('n_estimators', 100, 10100,step=250),
     "subsample":trial.suggest_categorical("subsample", [None]),
     "subsample_freq":trial.suggest_categorical("subsample_freq", [None]),
     "reg_alpha":trial.suggest_categorical("reg_alpha", [None]),
     "colsample_bytree":trial.suggest_categorical("colsample_bytree", [None]),
     "reg_lambda":trial.suggest_categorical("reg_lambda", [None]),
     #'class_weight':weights,
     "verbosity":trial.suggest_categorical("verbosity", [-1]),
         "n_jobs":trial.suggest_categorical("n_jobs", [-1]),
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
        iterations.append(best_iter)
        avg_number_iterations=(sum(iterations)/len(iterations))/(n_splits-1)*n_splits
    print (f' Average number of iterations in {trial.number} = {avg_number_iterations}')
    n=sum(results)/len(results)   
    return n

if  optuna_study == optuna_models[model_name]:
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


LGBM8_final_params={'device': method_LGBM, 'metric': LGBM_metrics[metric], 'lambda_l1': 0.6383608126975904, 'lambda_l2': 3.36855979995641, 'learning_rate': 0.028532322580273682, 'num_leaves': 67, 'feature_fraction': 0.6240926392648987, 'bagging_fraction': 0.869813676363038, 'early_stopping_round': 200, 'bagging_freq': 7, 'min_child_samples': 45, 'n_estimators': 3850, 'subsample': None, 'subsample_freq': None, 'reg_alpha': None, 'colsample_bytree': None, 'reg_lambda': None, 'verbosity': -1}


params_best_set=[[LGBM8_final_params,LGBM8_final_params]]


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
        print (f'k-fold result {round(sum(results)/len(results),k_fold_result_round)}')
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
        #model = lgb.LGBMRegressor(**params[1]).fit(X_reindex,y_reindex)
        #preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
        #preds_test[f'{model_name}_{z}_{j}']=model.predict(test)
        #k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)


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
        'n_estimators': trial.suggest_int('n_estimators', 100,10100,step=250),
        'max_depth': trial.suggest_categorical('max_depth', [2,3,4,5,6,7,8,9,10]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree',1,1),
        'early_stopping_rounds':trial.suggest_int('early_stopping_rounds',200,200),
       # 'scale_pos_weight':weight[0],
        'verbosity': trial.suggest_categorical('verbosity', [0]),
        }
    results=[]
    iterations=[]  
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = xgb.XGBRegressor(**param).fit(train_X,train_y,
                                      eval_set=[(valid_X,valid_y)],
                                      verbose=0)  
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        best_iter = model.best_iteration
        iterations.append(best_iter)
    avg_number_iterations=(sum(iterations)/len(iterations))/(n_splits-1)*n_splits
    print (f' Average number of iterations in {trial.number} = {avg_number_iterations}')
    n=sum(results)/len(results) 
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=10000)


XGB1_final_params={'device': method_XGB, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'lambda': 2.188811677466625, 'alpha': 0.47473786897623393, 'colsample_bytree': 0.7403597685071385, 'subsample': 0.9405176549242942, 'learning_rate': 0.011963990844615541, 'n_estimators': 6600, 'max_depth': 8, 'min_child_weight': 9, 'num_parallel_tree': 1, 'early_stopping_rounds': 200, 'verbosity': 0}


params_best_set=[[XGB1_final_params,XGB1_final_params],]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            model = xgb.XGBRegressor(**params[0]).fit(train_X,train_y,
                                              eval_set=[(train_X,train_y),(valid_X,valid_y)],
                                              verbose=0)  
            
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {round(sum(results)/len(results),k_fold_result_round)}')
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
        #model = xgb.XGBRegressor(**params[1]).fit(X_reindex,y_reindex)
        #k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)
        #preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
        #preds_test[f'{model_name}_{z}_{j}']=model.predict(test)


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
        "iterations":trial.suggest_int("iterations", 100, 10100,step=250),
        "learning_rate":trial.suggest_float("learning_rate", 0.001,0.1),
        "depth":trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        "bootstrap_type":trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        "random_strength":trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature":trial.suggest_float("bagging_temperature", 0, 10.0),
        "od_type":trial.suggest_categorical("od_type", ["Iter"]),
        "od_wait":trial.suggest_int("od_wait", 200, 200),
        "verbose":trial.suggest_categorical("verbose", [0]),
        #'scale_pos_weight':weight[1],
        }
    results=[]
    iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = CatBoostRegressor(**param).fit(train_X,train_y,
                                            eval_set=[(valid_X,valid_y)],
                                            verbose=0)  
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
        best_iter = model.get_best_iteration()
        iterations.append(best_iter)
    avg_number_iterations=(sum(iterations)/len(iterations))/(n_splits-1)*n_splits
    print (f' Average number of iterations in {trial.number} = {avg_number_iterations}')
    n=sum(results)/len(results)    
    return n

if  optuna_study==optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=10000)
    print('Best trial:', study.best_trial.params)


CAT4_final_params={'task_type': method_CAT, 'loss_function': 'RMSE', 'eval_metric': 'RMSE', 'use_best_model': True, 'iterations': 9350, 'learning_rate': 0.08088725075672568, 'depth': 7, 'l2_leaf_reg': 0.4078250147237912, 'bootstrap_type': 'Bayesian', 'random_strength': 0.0015714307144449541, 'bagging_temperature': 0.11052460727181501, 'od_type': 'Iter', 'od_wait': 200, 'verbose': 0}


params_best_set=[[CAT4_final_params,CAT4_final_params]]


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
        print (f'k-fold result {round(sum(results)/len(results),k_fold_result_round)}')
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)
       #model = CatBoostRegressor(**params[1]).fit(X_reindex,y_reindex)
       #k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)
       #preds_val.setdefault(f'{model_name}_{z}_{j}_full',[]).extend(model.predict(X_reindex))
       #preds_test[f'{model_name}_{z}_{j}']=model.predict(test)


model_name='ET'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        'n_estimators':trial.suggest_int("n_estimators", 10, 1010,step=25),
        'criterion':trial.suggest_categorical("criterion", [ET_metrics[metric]]), 
        'max_depth':trial.suggest_int("max_depth", 7, 20,step=1),        
        'min_samples_split':trial.suggest_int("min_samples_split", 2, 8,step=1),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 1, 8,step=1),        
        'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
        #'class_weight':trial.suggest_categorical("class_weight", [weights]),
        'verbose':trial.suggest_categorical("verbose", [0]),
        'max_features':trial.suggest_categorical("max_features", ['sqrt','log2']),
        'verbose':0}
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = ExtraTreesRegressor(**param).fit(train_X,train_y)  
        result = mean_squared_error(valid_y,(model.predict(valid_X)))**(1/2)
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


ET1_final_params={'n_estimators': 585, 'criterion': 'squared_error', 'max_depth': 17, 'min_samples_split': 7, 'min_samples_leaf': 1, 'n_jobs': -1, 'verbose': 0, 'max_features': 'sqrt'}


params_best_set=[[ET1_final_params,ET1_final_params]]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]

            model = ExtraTreesRegressor(**params[0]).fit(train_X,train_y)  
                              
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {round(sum(results)/len(results),k_fold_result_round)}')    
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)  
        #model = ExtraTreesRegressor(**params[1]).fit(X_reindex,y_reindex)
        #preds_test[f'{model_name}_{z}_{j}']=model.predict(test)
        #k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)


model_name='RF'
X = X_model
test = test_model
y=y_model


def objective(trial):   
    param = {
        'criterion':trial.suggest_categorical("criterion", [RF_metrics[metric]]), 
        'n_estimators':trial.suggest_int("n_estimators", 10, 1010,step=25),
        'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
        'verbose':trial.suggest_categorical("verbose", [0]),
        'max_depth':trial.suggest_int("max_depth", 3, 18,step=1),
        'min_samples_split':trial.suggest_int("min_samples_split", 2, 5,step=1),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 1, 5,step=1),
        'max_features':trial.suggest_categorical("max_features", ['sqrt','log2']),
        'warm_start':trial.suggest_categorical("warm_start", [True]),
       #'class_weight':trial.suggest_categorical("class_weight", [weights]),
             }
    results=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestRegressor(**param).fit(train_X,train_y)  
        result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
        results.append(result)
    n=sum(results)/len(results)        
    return n

if optuna_study == optuna_models[model_name]:
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


RF1_final_params={'criterion': 'squared_error', 'n_estimators': 935, 'n_jobs': -1, 'verbose': 0, 'max_depth': 14, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'warm_start': True}


params_best_set=[[RF1_final_params,RF1_final_params]]


if finish_models[model_name] and finish_set:
    for z,params in enumerate(params_best_set):
        
        results=[]
        preds_test_temp={}
        for i,(train_index, test_index) in enumerate(cv.split(X, y)):
            j=i//n_splits
            start_time = time.time()
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        
            model = RandomForestRegressor(**params[0]).fit(train_X,train_y)  
                                         
            preds_val.setdefault(f'{model_name}_{z}_{j}',[]).extend(model.predict(valid_X))
            preds_test_temp.setdefault(f'{model_name}_{i}',[]).extend(model.predict(test))
           
            results.append(metric_call())
        print (f'k-fold result {round(sum(results)/len(results),k_fold_result_round)}')      
        preds_test[f'{model_name}_{z}_{j}']=pd.DataFrame.from_dict(preds_test_temp).mean(axis=1)    
        #model = RandomForestRegressor(**params[1]).fit(X_reindex,y_reindex)
        #preds_test[f'{model_name}_{z}_{j}']=model.predict(test)
        #k=mean_squared_error(y_reindex,model.predict(X_reindex))**(1/2)


model_name='LCV'
X = X_model
test = test_model
y=y_model


def objective(trial):

    param = {
        'precompute':trial.suggest_categorical("precompute", ['auto']),
        'fit_intercept':trial.suggest_categorical("fit_intercept", [True]), 
        'max_iter':trial.suggest_int("max_iter", 10, 1010,step=25),
        'verbose':trial.suggest_categorical("verbose", [False]),
        "eps":trial.suggest_float("eps", 1e-04, 100.0, log=True),
        'n_alphas':trial.suggest_int("n_alphas", 100, 1000,step=25),
        'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
             }

    results=[]
    n_iterations=[]
    for train_index, test_index in cv.split(X, y):
        train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
        train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
        model = LassoCV(**param).fit(train_X,train_y,
                                           # eval_set=[(valid_X,valid_y)],
                                          #  verbose=0
                                             )  
        result = mean_squared_error(valid_y,((model.predict(valid_X))))**(1/2)
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

reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error', 
                                                      factor=0.8, 
                                                      patience=3, 
                                                      verbose=0,
                                                      min_delta=0.0001 ,
                                                      min_lr=0.000001, 
                                                      mode='max')

early_stopping = keras.callbacks.EarlyStopping(patience=8,
                                               verbose=0,
                                               min_delta=0.0001,
                                               monitor="mean_squared_error",
                                               mode='min',
                                               restore_best_weights=True)

def get_model(trial):
    opt = tf.optimizers.Adam(learning_rate=0.001)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error', 
                                                          factor=0.8, 
                                                          patience=3, 
                                                          verbose=0,
                                                          min_delta=0.0001 ,
                                                          min_lr=0.000001, 
                                                          mode='min')

    early_stopping = keras.callbacks.EarlyStopping(patience=8,
                                                   verbose=0,
                                                   min_delta=0.0001,
                                                   monitor="mean_squared_error",
                                                   mode='min',
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
    loss=tf.losses.SigmoidFocalCrossEntropy(alpha=0.80, gamma=2.0),
    #loss='binary_crossentropy',
    metrics=['mean_squared_error'])  
    weights_before_update=model.get_weights()
    
    preds=[]
    results=[]
    i=0
    
    for train_index, test_index in (cv.split(X, y)):
        opt = tf.optimizers.Adam(learning_rate=0.001)
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error', 
                                                              factor=0.8, 
                                                              patience=5, 
                                                              verbose=0,
                                                              min_delta=0.0001 ,
                                                              min_lr=0.000001, mode='max')

        early_stopping = keras.callbacks.EarlyStopping(patience=10,
                                                       min_delta=0.0001,
                                                       monitor="mean_squared_error",
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
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError(),
    ])  
    
    return model


if finish_models[model_name] == finish_set:
    preds=[]
    results=[]
    predicts=pd.DataFrame()
    
    
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        opt = tf.optimizers.Adam(learning_rate=0.1)
        
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                              factor=0.5, 
                                                              patience=5, 
                                                              verbose=1,
                                                              min_delta=0.0001 ,
                                                              min_lr=0.000001, mode='min')

        early_stopping = keras.callbacks.EarlyStopping(patience=30,
                                                       min_delta=0.0001,
                                                       monitor='val_loss',
                                                       mode='min',
                                                    #   restore_best_weights=True,
                                                       verbose=1,
                                                      )

        start_time = time.time()
        
        train_X, valid_X = X[train_index], X[test_index]
        train_y, valid_y = y[train_index], y[test_index]
        
        model = get_model()
        
        history = model.fit(
            train_X,train_y,
            validation_data=(valid_X,valid_y),
            batch_size=512*10,
            epochs=200,
            #class_weight = weights,
            callbacks=[early_stopping,reduce_lr_loss],
            verbose=1)

        
        pred = (model.predict(valid_X))
        pred_test = (model.predict(test))
        preds.append(pred)
        
        predicts[i]=pd.DataFrame(pred_test)
        
        preds_val[f'{model_name}_{i}_{j}']=model.predict(valid_X).reshape(1,-1)[0]
        preds_test[f'{model_name}_{i}_{j}']=model.predict(test).reshape(1,-1)[0]
        
        result = round(mean_squared_error(valid_y,pred)**(1/2),6)
        
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
    
    model = H2OAutoML(max_runtime_secs=60*60*10, seed = 3)
    model.train(x = list(X.columns), y = target, training_frame = h2o_train)
    
    output_path = "/kaggle/working/H2Oensemble.csv" 
    h2o_sample[target]=model.leader.predict(h2o_test)
    h2o.export_file(h2o_sample, path=output_path, force=True)


X = X_reindex
test = test_model
y=y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


CV_models_results=pd.DataFrame()
CV_models_results['models']=list(X_ensemble.columns)
CV_result=[]
for i,n in enumerate(list(X_ensemble.columns)):
    CV_result.append(round(mean_squared_error(X_ensemble[n],y)**(1/2),6))
CV_models_results['results']=CV_result
CV_models_results.sort_values(by='results')


for n in test_ensemble:
    sample[target]=test_ensemble[n]
    sample.to_csv(f'{n}.csv', index=False)


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


if ensemble_models[model_name] and ensemble_set:
    model = FlamlAutoML()
    model.fit(X_train=X_ensemble, 
               y_train = y, 
               task="regression",
               n_jobs=4,
               metric='rmse',
               time_budget=60*60*4,
               verbose=3,
               ensemble=True,
               eval_method='cv',
               n_splits =6,
               use_spark  = False,
               seed=1
              )
    sample[target]=model.predict(test_ensemble)
    sample.to_csv('FLAML_Ensemble.csv', index=False)


################
### Settings ###
################
model_name='WE_Ensemble'

X = X_reindex
test = test_model

y=y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


WE_cols=list(X_ensemble.columns)

if ensemble_models[model_name] and ensemble_set:
    def objective(trial):
        WE_key_names={}
        for n in WE_cols:
            WE_key_names["k_"+n]=trial.suggest_float("k_"+n, 0.0, 1,step=0.001)
        WE_key_val=np.array(list(WE_key_names.values()))
        WE_key_val=WE_key_val/sum(WE_key_val)
        result=mean_squared_error((X_ensemble*WE_key_val).sum(axis=1),y)**(1/2)
        return result

    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                        direction=direction)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=10000)
   
    print('Best trial:', round(study.best_trial.values[0],6))
    print('Best trial:', study.best_trial.params)
    final_weights=np.array(list(study.best_trial.params.values()))/sum(np.array(list(study.best_trial.params.values())))
    sample[target]= (test_ensemble*final_weights).sum(axis=1)
    sample.to_csv('WE_Ensemble.csv', index=False)


################
### Settings ###
################
model_name='RidgeCV_Ensemble'

X = X_reindex
test = test_model

y=y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


optuna.logging.set_verbosity(optuna.logging.INFO)
if ensemble_models[model_name] and ensemble_set:
    def objective(trial):
        start_time = time.time()
        param = {'alphas':[1, 10,100, 1000]}
                      
        results=[]
        iterations=[]  
        for train_index, test_index in cv.split(X_ensemble, y):
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            model = RidgeCV(**param).fit(train_X,train_y)  
            result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
            results.append(result)
           # best_iter = model.best_iteration
           # iterations.append(best_iter)
       # avg_number_iterations=(sum(iterations)/len(iterations))/(n_splits-1)*n_splits
       # print (f'Average number of iterations in trial#{trial.number} = {avg_number_iterations}')
        print (f'Time in trial#{trial.number} is {np.round((time.time()-start_time),1)}')
        n=sum(results)/len(results) 
        return n
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=10)
     
    print('Best trial:', round(study.best_trial.values[0],6))
    print('Best trial:', study.best_trial.params)

    sample[target]= (test_ensemble*final_weights).sum(axis=1)
    sample.to_csv('RidgeCV_Ensemble.csv', index=False)


################
### Settings ###
################
model_name='LR_Ensemble'
X = X_reindex
test = test_model
y = y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


if ensemble_models[model_name] and ensemble_set:
    def objective(trial):
        print("START__________________________________")       
        param = {
           #'precompute':"auto",
           # 'fit_intercept':True,
           # 'normalize':False,
            #'max_iter':1000,
            #'verbose':False,
            'copy_X': True, 
                   'fit_intercept': True, 
                   'n_jobs': 10,
                   'positive': False,
            #'eps':1e-04,
           # 'cv':cv,
            #'n_alphas':1000,
            #'task_type':method,
           # "#loss_function":trial.suggest_categorical("loss_function", ["Logloss"]),
            
           # 'penalty':trial.suggest_categorical("penalty", ['l1', 'l2', 'elasticnet', 'none']) ,
           # 'solver':trial.suggest_categorical("solver", ['lbfgs','newton-cg','liblinear','sag','saga']) ,
           # "max_iter":trial.suggest_int("max_iter", 100, 4000), 
            #"C":trial.suggest_float("C", 1e-4, 4, log=True),
            #'class_weight':{0:1,1:1}
        }
            
        results=[]
        for train_index, test_index in cv.split(X_ensemble, y):
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            model = LinearRegression(**param).fit(train_X,train_y)
            result = mean_squared_error(valid_y,((model.predict(valid_X))))**(1/2)
            results.append(result)
        n=sum(results)/len(results)       
        return n
    
    optuna_study==optuna_models['LR']
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=1000)
    print('Best trial:', study.best_trial.params)


################
### Settings ###
################
model_name='KERAS'
X = X_reindex
test = test_model
y = y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


X =X_ensemble.to_numpy()
test = test_ensemble.to_numpy()
y = y_reindex.to_numpy()
input_shape = [X.shape[1]]


opt = tf.optimizers.Adam(learning_rate=0.001)


def get_model():
    model = keras.Sequential([
            layers.BatchNormalization(input_shape = [X.shape[1]]),
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
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError(),
    ])  
    
    return model


if finish_models[model_name] == finish_set:
    preds=[]
    results=[]
    predicts=pd.DataFrame()
    
    
    for i,(train_index, test_index) in enumerate(cv.split(X, y)):
        opt = tf.optimizers.Adam(learning_rate=0.1)
        
        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                              factor=0.5, 
                                                              patience=5, 
                                                              verbose=1,
                                                              min_delta=0.0001 ,
                                                              min_lr=0.000001, mode='min')

        early_stopping = keras.callbacks.EarlyStopping(patience=30,
                                                       min_delta=0.0001,
                                                       monitor='val_loss',
                                                       mode='min',
                                                    #   restore_best_weights=True,
                                                       verbose=1,
                                                      )

        start_time = time.time()
        
        train_X, valid_X = X[train_index], X[test_index]
        train_y, valid_y = y[train_index], y[test_index]
        
        model = get_model()
        
        history = model.fit(
            train_X,train_y,
            validation_data=(valid_X,valid_y),
            batch_size=512*10,
            epochs=200,
            #class_weight = weights,
            callbacks=[early_stopping,reduce_lr_loss],
            verbose=1)

        
        pred = (model.predict(valid_X))
        pred_test = (model.predict(test))
        preds.append(pred)
        
        predicts[i]=pd.DataFrame(pred_test)
        
        preds_val[f'{model_name}_{i}_{j}']=model.predict(valid_X).reshape(1,-1)[0]
        preds_test[f'{model_name}_{i}_{j}']=model.predict(test).reshape(1,-1)[0]
        
        result = round(mean_squared_error(valid_y,pred)**(1/2),6)
        
        print (f'\033[0;33;40m Step #{i}.' + f"--- {time.time() - start_time}s sec ---" + f"Auc result = {result} \033[0;30;0m")
        results.append(result)
    print (f'\033[0;35;40m Final KERAS result = {(sum(results)/len(results))} \033[0;30;0m')


################
### Settings ###
################
model_name='XGB_Ensemble'
X = X_reindex
test = test_model
y = y_reindex

X_ensemble=pd.DataFrame.from_dict(preds_val)
test_ensemble=pd.DataFrame.from_dict(preds_test)


optuna.logging.set_verbosity(optuna.logging.INFO)
if ensemble_models[model_name] and ensemble_set:
    def objective(trial):
        start_time = time.time()
        param = {  
            'device':trial.suggest_categorical('device',[method_XGB]),
            'objective': trial.suggest_categorical('objective',[XGB_tasks[task]]),
            'eval_metric': trial.suggest_categorical('eval_metric',[XGB_metrics[metric]]),
            'lambda': trial.suggest_float('lambda', 0, 10.0),
            'alpha': trial.suggest_float('alpha', 0, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1,1.0),
            'subsample': trial.suggest_float('subsample', 0.2,1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01,0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100,2100,step=250),
            'max_depth': trial.suggest_categorical('max_depth', [2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'num_parallel_tree': trial.suggest_int('num_parallel_tree',1,1),
            'early_stopping_rounds':trial.suggest_int('early_stopping_rounds',200,200),
           # 'scale_pos_weight':weight[0],
            'verbosity': trial.suggest_categorical('verbosity', [2]),
            }
        results=[]
        iterations=[]  
        for train_index, test_index in cv.split(X_ensemble, y):
            train_X, valid_X = X.iloc[train_index], X.iloc[test_index]
            train_y, valid_y = y.iloc[train_index], y.iloc[test_index]
            model = xgb.XGBRegressor(**param).fit(train_X,train_y,
                                          eval_set=[(valid_X,valid_y)],
                                          verbose=0)  
            result = mean_squared_error(valid_y,model.predict(valid_X))**(1/2)
            results.append(result)
            best_iter = model.best_iteration
            iterations.append(best_iter)
        avg_number_iterations=(sum(iterations)/len(iterations))/(n_splits-1)*n_splits
        print (f'Average number of iterations in trial#{trial.number} = {avg_number_iterations}')
        print (f'Time in trial#{trial.number} is {np.round((time.time()-start_time),1)}')
        n=sum(results)/len(results) 
        return n
    study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(),
                                direction=direction)
    study.optimize(objective, n_trials=200)
     
    print('Best trial:', round(study.best_trial.values[0],6))
    print('Best trial:', study.best_trial.params)


print('finish')

