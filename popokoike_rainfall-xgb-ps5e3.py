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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix,classification_report

import xgboost as xgb

import optuna


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head()


test.head()


submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print('train:',train.shape)
print('test:',test.shape)
print('submission:',submission.shape)


train.isnull().sum()


test.isnull().sum()


test['winddirection']=test['winddirection'].fillna(train['winddirection'].mean())


test.isnull().sum()


target='rainfall'
features=train.loc[:,:'windspeed'].columns.tolist()
features.remove('id')
features


X=train[features].copy()
y=train[target].copy()


base_params={
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'booster':'gbtree',
    'seed':42,
}


# kf=StratifiedKFold(n_splits=5,
#                    shuffle=True,
#                    random_state=42)
n_splits=5
tscv=TimeSeriesSplit(n_splits=n_splits)

def objective(trial):
    params={
        'eta':trial.suggest_float('eta',1e-3,1e-1,log=True),
        
        'max_depth':trial.suggest_int('max_depth',3,18),
        
        'min_child_weight':trial.suggest_int('min_child_weight',1,1000,log=True),
        
        'subsample':trial.suggest_float('subsample',0.3,1.0),
        
        'colsample_bytree':trial.suggest_float('colsample_bytree',0.3,1.0),
        
        'alpha':trial.suggest_float('alpha',1e-5,1e1,log=True),
        
        'lambda':trial.suggest_float('lambda',1e-5,1e1,log=True),
    }
    
    params.update(base_params)
    
    auc_scores=[]
    
    for trn_idx,val_idx in tscv.split(X,y):
        X_trn,X_val=X.iloc[trn_idx],X.iloc[val_idx]
        y_trn,y_val=y.iloc[trn_idx],y.iloc[val_idx]
        
        
        dtrain=xgb.DMatrix(X_trn,label=y_trn)
        dval=xgb.DMatrix(X_val,label=y_val)
        
        
        model=xgb.train(params=params,
                        dtrain=dtrain,
                        num_boost_round=5000,
                        early_stopping_rounds=100,
                        evals=[(dval,'validation')],
                        verbose_eval=False)
        
        
        pred_proba=model.predict(dval)
        
        
        auc=roc_auc_score(y_val,pred_proba)
        auc_scores.append(auc)
    return np.mean(auc_scores)


# optuna.delete_study(study_name='xgb_3_tscv_5',
#                     storage='sqlite:///example.db')


study=optuna.create_study(study_name='xgb_3_tscv_5',
                          storage='sqlite:///example.db',
                          direction='maximize',
                          load_if_exists=True,
                          sampler=optuna.samplers.TPESampler(seed=42,
                                                             n_startup_trials=40,
                                                             multivariate=True))
study.optimize(objective,
               n_trials=60,
               n_jobs=1)


print('Best trial:')
trial=study.best_trial
print(f'Best hyperparameters: {trial.params}')
print(f'Best value: {trial.value}')



best_params=trial.params


best_params.update(base_params)

# kf=StratifiedKFold(n_splits=7,
#                    shuffle=True,
#                    random_state=42)

tscv=TimeSeriesSplit(n_splits=n_splits)


oof_preds=np.zeros(train.shape[0])

test_preds=np.zeros(test.shape[0])

for trn_idx,val_idx in tscv.split(X,y):
    X_trn,X_val=X.iloc[trn_idx],X.iloc[val_idx]
    y_trn,y_val=y.iloc[trn_idx],y.iloc[val_idx]
    
    
    dtrain=xgb.DMatrix(X_trn,label=y_trn)
    dval=xgb.DMatrix(X_val,label=y_val)
    dtest=xgb.DMatrix(test[features])
    
    model=xgb.train(params=best_params,
                    dtrain=dtrain,
                    num_boost_round=5000,
                    early_stopping_rounds=100,
                    evals=[(dval,'validation')],
                    verbose_eval=False)
    
    
    oof_preds[val_idx]=model.predict(dval)
    test_preds+=model.predict(dtest)

test_pred=test_preds/tscv.n_splits


auc=roc_auc_score(y,oof_preds)


print(f'AUC: {auc}')


print(classification_report(y,np.where(oof_preds>0.5,1,0)))


confusion_matrix(y,np.where(oof_preds>0.5,1,0))


submission[target]=test_pred
submission.to_csv('submission.csv',index=False)
print(submission.shape)
submission.head()


importance=model.get_score(importance_type='gain')
importance=pd.DataFrame({'feature':list(importance.keys()),
                         'gain':list(importance.values())})
importance=importance.sort_values('gain',ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(data=importance.head(50),
            x='gain',
            y='feature')

