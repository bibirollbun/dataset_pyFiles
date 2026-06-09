import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt 
import xgboost as xgb #use xgb as algorthm model
from xgboost import XGBRegressor #use xgb regressor because we want to predict continuous numerical value
from sklearn.preprocessing import LabelEncoder #encode all categorical and boolean feature
import optuna # hyperparameter tuning using optuna
from sklearn.model_selection import cross_val_score #to evaluate model
from sklearn.model_selection import StratifiedKFold #to perform k-fold


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



data_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
data_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


#divide dataset into 4
y_train = data_train['accident_risk'] #target train
train_ids = data_train['id'] #target train id
test_ids = data_test['id'] #target test id
train = data_train.drop(['accident_risk','id'],axis=1) #train feature 
test = data_test.drop(['id'],axis=1) # test feature


X_train = train.copy()
X_test = test.copy()


#feature encoding

#first select categorical,object and boolean column
cat_col = X_train.select_dtypes(['object','bool']).columns


#label encode

#in test data we use only transform, to prevent inconsistency, like example: in training blue encoded as 1,
#so when in test dataset the blue also encoded as 1 not 0 or other number.
for col in cat_col:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



def objective(trial):

    #optuna will search the best output given a specified range of numbers
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_bin': trial.suggest_int('max_bin',100,600),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.1),
        'max_depth': trial.suggest_int('max_depth',1,10),
        'min_child_weight': trial.suggest_int('min_child_weight',1,10),
        'subsample': trial.suggest_float('subsample',0.1,1),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.1,1),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel',0.1,1),
        'colsample_bynode': trial.suggest_float('colsample_bynode',0.1,1),
        'reg_alpha': trial.suggest_float('reg_alpha',0.1,1),
        'reg_lambda': trial.suggest_float('reg_lambda',0.1,1),
        'gamma': trial.suggest_float('gamma',0.1,1),
        'max_delta_step': trial.suggest_int('max_delta_step',0,10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight',0.1,1)
    }

    model = XGBRegressor(**xgb_params,tree_method='hist',device='cpu') #configure xgbregressor model

    #evaluate model
    score = cross_val_score(model,X_train,y_train,cv=7,scoring='neg_mean_absolute_error')#cross_val_score do .fit() internally
    mae = -score.mean()
    return mae



#configure optuna
study = optuna.create_study(study_name='xgboost_optuna_roadaccident_comp',direction='minimize')
study.optimize(objective,n_trials=100,show_progress_bar=True,n_jobs=-1) #njobs -1 to run on all cores to speed up process



#use 7 fold
FOLD = 7


#stratification bins
#In practice, many data scientists use standard K-fold for regression first, 
# and only implement binned stratification if the results show high variance across folds 
# or if the target distribution is highly imbalanced.
#q=10 best practice, and must be q > k.(q  = 10 must be larger than k = 7 fold)
#this q will also have positive relationship with data,if data is large q also should be increased

y_bins = pd.qcut(y_train,q=10,labels=False,duplicates='drop')


skf = StratifiedKFold(n_splits=FOLD,shuffle=True,random_state=42)
fold_splits = list(skf.split(X_train,y_bins))


oof_prediction = np.zeros(len(X_train))
test_prediction = np.zeros(len(X_test))

fold_scores = []
# feature_importance_dict = {}


for fold,(train_idx,val_idx) in enumerate(fold_splits,1):
    print(f'Fold {fold}/{FOLD}')

    X_tr,X_val = X_train.iloc[train_idx],X_train.iloc[val_idx]
    y_tr,y_val = y_train.iloc[train_idx],y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr,label=y_tr)
    dval = xgb.DMatrix(X_val,label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        params = study.best_params,
        dtrain = dtrain,
        num_boost_round = 10000,
        evals = [(dval,'valid')],
        early_stopping_rounds = 200,
        verbose_eval = False
    )

    oof_prediction[val_idx] = model.predict(dval)
    test_prediction += model.predict(dtest)

    fold_rmse = np.sqrt(np.mean((oof_prediction[val_idx] - y_val)**2))
    fold_scores.append(fold_rmse)
    print(f'FOLD {fold} RMSE:{fold_rmse:.6f}')

test_prediction/=FOLD


print(test_prediction)

