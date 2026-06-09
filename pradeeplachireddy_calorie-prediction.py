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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv',index_col='id').reset_index(drop=True)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv',index_col='id').reset_index(drop=True)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


df_train.head(2)


df_test.head(2)


df_train.columns


df_train.info()


df_test.info()


df_train['Sex'] = df_train['Sex'].map({'male':0,'female':1})
df_test['Sex'] = df_test['Sex'].map({'male':0,'female':1})


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_log_error,make_scorer
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from cuml.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


df_train = df_train.sample(frac=1,random_state=42).reset_index(drop=True)


has_neg = np.any(df_train['Calories']<0)
print('negative?',has_neg)


df_train.head(2)


df_train.head(2)


y = df_train['Calories'].copy()
X = df_train.drop(columns=['Calories'],axis=1)


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


def safe_rmsle(y_true,y_pred):
    y_pred = np.maximum(y_pred,0) # or use np.clip(val,0,None)
    y_true = np.maximum(y_true,0)
    return np.sqrt(mean_squared_log_error(y_true,y_pred))
rmsle_scorer=make_scorer(safe_rmsle,greater_is_better=False)


cv=KFold(n_splits=10,shuffle=True, random_state=42)


LR = LinearRegression()
param_grid_lr = {} # no hyper params for basic linear regression
GCV_lr = GridSearchCV(LR,param_grid=param_grid_lr,cv=cv,scoring=rmsle_scorer,n_jobs=-1,verbose=1,error_score='raise')
# for classification use accuracy and for regression use
#mse,rmse,mae, but in gridsearch, you have to use neg_mean_squared_error like this for other also
GCV_lr.fit(X_train,y_train)


# rf = RandomForestRegressor(random_state=42)
# param_grid_rf = {'n_estimators':[100,200],'max_depth':[None,5,10],'min_samples_split':[2,5],'min_samples_leaf':[1,2]}
# GCV_rf = GridSearchCV(rf,param_grid_rf,cv=cv,scoring=rmsle_scorer,n_jobs=-1,verbose=1,error_score='raise')
# GCV_rf.fit(X_train,y_train)


xgb = XGBRegressor(tree_method = "hist", device = "cuda",random_state=42)
param_grid_xgb = {'n_estimators':[100,200],'max_depth':[3,5,7],'learning_rate':[0.01,0.1,0.2]}
GCV_xgb = GridSearchCV(xgb,param_grid_xgb,cv=cv,scoring=rmsle_scorer,n_jobs=-1,verbose=1,error_score='raise')
GCV_xgb.fit(X_train,y_train)


!pip install optuna
import optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "tree_method":"hist",
        "device":"cuda",
        "random_state": 42,
    }
    model = XGBRegressor(**params, n_jobs=-1)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # rmsle_scores=[]
    # for train_idx,val_idx in kf.split():
    #     X_train,X_val = X[train_idx],X[val_idx]
    #     y_train,y_val = y[train_idx],y[val_idx]
    #     model.fit(X_train,y_train)
    #     y_pred = model.predict(X_val)
    #     score = rmsle(y_val,y_pred)
    scores = cross_val_score(model, X_train,y_train, scoring=rmsle_scorer, cv=kf)
    print(scores)
    return scores.mean()
    
    


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)
print("Best trial:")
print(study.best_trial)



best_params_optuna = study.best_trial.params
best_params_optuna


xgb_bayes = XGBRegressor(**best_params_optuna,tree_method = "hist", device = "cuda",random_state=42)
xgb_bayes.fit(X_train,y_train)


best_lr = GCV_lr.best_estimator_
# best_rf = GCV_rf.best_estimator_
best_xgb = GCV_xgb.best_estimator_



model_accuracy = {}


pred_lr = best_lr.predict(X_val)
# pred_rf = best_rf.predict(X_val)
pred_xgb = best_xgb.predict(X_val)
pred_xgb_bayes = xgb_bayes.predict(X_val)
pred_lr = np.maximum(pred_lr,0)
# pred_rf = np.maximum(pred_rf,0)
pred_xgb = np.maximum(pred_xgb,0)
pred_xgb_bayes = np.maximum(pred_xgb_bayes,0)


accuracy_lr = mean_squared_log_error(y_val,pred_lr)
# accuracy_rf = mean_squared_log_error(y_val,pred_rf)
accuracy_xgb = mean_squared_log_error(y_val,pred_xgb)
accuracy_xgb_bayes = mean_squared_log_error(y_val,pred_xgb_bayes)

model_accuracy[best_lr]=np.sqrt(accuracy_lr)
# model_accuracy[best_rf]=np.sqrt(accuracy_rf)
model_accuracy[best_xgb]=np.sqrt(accuracy_xgb)
model_accuracy[xgb_bayes]=np.sqrt(accuracy_xgb_bayes)

model_accuracy


best_model = min(model_accuracy,key=model_accuracy.get)
best_model


test_pred = best_model.predict(df_test)


has_neg = np.sum(test_pred<0)
print('negative?',has_neg)



test_pred = np.maximum(test_pred,0)
test_pred


has_neg = np.sum(test_pred<0)
print('negative?',has_neg)


len(test_pred)


sample_submission['Calories']=test_pred
sample_submission.to_csv('sample_submission.csv',index=False)


sample_submission.head()


# lets train model on full data using best model instead of splitting for validation
# this way model will be trained on more data and we may get better test accuracy
# this is common practice - after evaludating val data we will train again using full data


best_params_lr = GCV_lr.best_params_
best_lr = LinearRegression(**best_params_lr)
best_lr.fit(X,y)
test_pred_lr = best_lr.predict(df_test)
test_pred_lr = np.maximum(test_pred_lr,0)


submission_lr =pd.DataFrame({'id':sample_submission['id'],'Calories':test_pred_lr})
submission_lr.to_csv('submission_lr.csv',index=False)


# best_params_rf = GCV_rf.best_params_
# best_rf = RandomForestRegressor(**best_params_rf)
# best_rf.fit(X,y)
# test_pred_rf = best_rf.predict(df_test)
# test_pred_rf = np.maximum(test_pred_rf,0)


# submission_rf =pd.DataFrame({'id':sample_submission['id'],'Calories':test_pred_rf})
# submission_rf.to_csv('submission_rf.csv',index=False)


best_params_xgb = GCV_xgb.best_params_
best_xgb = XGBRegressor(**best_params_xgb,tree_method = "hist", device = "cuda",random_state=42)
best_xgb.fit(X,y)
test_pred_xgb = best_xgb.predict(df_test)
test_pred_xgb = np.maximum(test_pred_xgb,0)


submission_xgb =pd.DataFrame({'id':sample_submission['id'],'Calories':test_pred_xgb})
submission_xgb.to_csv('submission_xgb.csv',index=False)


best_params_optuna_previous = {'n_estimators': 520,
 'max_depth': 10,
 'learning_rate': 0.024915517750350627,
 'subsample': 0.8679465936467534,
 'colsample_bytree': 0.8717536206508927,
 'reg_alpha': 0.745681383446576,
 'reg_lambda': 0.5340572643230692}


xgb_bayes_2 = XGBRegressor(**best_params_optuna_previous,tree_method = "hist", device = "cuda",random_state=42)
xgb_bayes_2.fit(X,y)
test_pred_xgb_bayes = xgb_bayes_2.predict(df_test)
test_pred_xgb_bayes = np.maximum(test_pred_xgb_bayes,0)


submission_xgb_bayes =pd.DataFrame({'id':sample_submission['id'],'Calories':test_pred_xgb_bayes})
submission_xgb_bayes.to_csv('submission_xgb_bayes.csv',index=False)










