import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from pandas.errors import PerformanceWarning
from sklearn.metrics import mean_squared_error
from itertools import combinations
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import copy


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
target = 'BeatsPerMinute'
cat_cols = []
num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
            'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']
log1p_cols = []
for i in range(len(num_cols)):
    for j in range(i, len(num_cols)):
        col1 = num_cols[i]
        col2 = num_cols[j]
        # Create interaction features
        train[f'{col1}_x_{col2}'] = train[col1] * train[col2]
        test[f'{col1}_x_{col2}'] = test[col1] * test[col2]
        train[f'{col1}_squared']= train[col1] * train[col1]
        test[f'{col1}_squared']= test[col1] * test[col1]
        # Create ratio features, handle division by zero
        if col1 != col2:
            train[f'{col1}_div_{col2}'] = train[col1] / (train[col2] + 1e-6)
            test[f'{col1}_div_{col2}'] = test[col1] / (test[col2] + 1e-6)
for col in num_cols:
    train[f'log1p_{col}'] = np.log1p( train[col] )
    test[f'log1p_{col}'] = np.log1p( test[col] )
    log1p_cols.append(f'log1p_{col}')

for col in log1p_cols:
    print(col, train[col].min(), train[col].max(), train[col].nunique())
def add_bins(df, column, labels, new_column=None):
    if len(labels) == 4 and new_column is None:
        new_column = f"{column}_quartile"
    if len(labels) == 5 and new_column is None:
        new_column = f"{column}_quintile"
    if len(labels) == 7 and new_column is None:
        new_column = f"{column}_septile"
    if len(labels) == 10 and new_column is None:
        new_column = f"{column}_decile"
    

    df[new_column] = pd.cut(
        df[column],
        bins= len(labels),
        labels=labels,
        include_lowest=True
    )
    return df[new_column]

Features = train.columns.tolist()
Features.remove(target)
Features.remove('id')

CATS = []
for col in num_cols:
    train[f"{col}_septile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7'])
    CATS.append(f"{col}_septile")
    
for col in num_cols:
    train[f"{col}_decile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                                   'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])
    CATS.append(f"{col}_decile")
for col in num_cols:
    test[f"{col}_septile"] = add_bins(test, col,  ['Q1', 'Q2', 'Q3', 'Q4','Q5','Q6','Q7'])
    
for col in num_cols:
    test[f"{col}_decile"] = add_bins(test, col, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                                                   'Q6', 'Q7', 'Q8', 'Q9', 'Q10'])
log1p_cols.remove('log1p_AudioLoudness')
for col in log1p_cols:
    train[f"{col}_quintile"] = add_bins(train, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5'])
    test[f"{col}_quintile"] = add_bins(test, col, ['Q1', 'Q2', 'Q3', 'Q4','Q5'])
    CATS.append(f"{col}_quintile")

Cat_groupby=[]
for col in CATS:
    new_col = f'mean_by_{col}'
    mapping = train.groupby(col)['BeatsPerMinute'].mean()
    
    # Apply to train and test
    train[new_col] = train[col].map(mapping)
    test[new_col] = test[col].map(mapping)
    Cat_groupby.append(new_col)
Features = Features + Cat_groupby
cat_features = CATS
X_cat =train[cat_features]
print(Features)
X=train[Features]
y=train[target]
X_test = test[Features]
display(train.shape)

xgb_params = {
    'n_estimators': 620,         
    'max_leaves': 211,            
    'min_child_weight': 1.5,     
    'max_depth': 6,               
    'grow_policy': 'lossguide',   
    'learning_rate': 0.0021858703356597603,      
    'tree_method': 'hist',        
    'subsample': 0.85,            
    'colsample_bylevel': 0.6787051322531533,     
    'colsample_bytree': 0.6843905004927857,       
    'colsample_bynode': 0.442116057736592,     
    'sampling_method': 'uniform',  
    'reg_alpha': 2.5,             
    'reg_lambda': 0.8,            
    'enable_categorical': True,    
    'max_cat_to_onehot': 1,       
    'device': 'cuda',            
    'n_jobs': -1,                 
    'random_state': 0,     
    'verbosity': 0,               
}
lgbm_params = {
    'learning_rate': 0.001502328415098844,
    'num_leaves': 79, 
    'max_depth': 14,
    'feature_fraction': 0.8933016300882094,
    'bagging_fraction': 0.9754103048412501,
    'bagging_freq': 7, 
    'min_child_samples': 40,
    'enable_categorical': True,   
    'lambda_l1': 7.10897934678165e-07,
    'lambda_l2': 7.81564014894075e-08,
    'random_state' : 0,
    'n_jobs' : -1,
    'verbosity': -1,
    'n_estimators': 643
}


cb_model=CatBoostRegressor(
    border_count= 28,
    colsample_bylevel= 0.19459088572914465,
    depth= 5,
    iterations= 600,
    l2_leaf_reg= 31.236169478676036,
    learning_rate= 0.1332583504067626,
    min_child_samples= 189,
    random_state= 0,
    random_strength= 0.8517786189616939,
    subsample= 0.3192330024411618,
    verbose= False,
    cat_features = CATS)
print(cat_features)
print('CATBOOST')
print()
cb_oof_preds = np.zeros(len(X))
cb_models, cb_scores=[],[]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in kf.split(X_cat, y):
        print('Fold:', len(cb_models) + 1)
        X_train, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cb_model.fit(X_train, y_train)
        cb_oof_preds[val_idx] = cb_model.predict(X_val)
        acc = mean_squared_error(y_val, cb_model.predict(X_val), squared=False)
        cb_scores.append(acc), cb_models.append(cb_model)
        print('Accuracy:', acc)
print('CATBOOST ACCURACY: ', np.mean(cb_scores))


def cb_objective(trial):
    # Suggest weights for the ensemble
    learning_rate=trial.suggest_float("learning_rate", 0.001, 0.12, log=True)
    depth = trial.suggest_int("depth", 2, 10)
    iterations = trial.suggest_int("iterations", 200, 2000)

    cb=CatBoostRegressor(learning_rate=learning_rate,depth=depth,iterations=iterations,random_state=0,verbose = False,
                        cat_features = CATS)
    X_train, X_val, y_train,y_val=train_test_split(X_cat,y,test_size=0.25,random_state=0)
    cb.fit(X_train, y_train)
    preds=cb.predict(X_val)
    score = mean_squared_error(y_val, preds, squared=False)
    return score

#cb_study = optuna.create_study(direction="minimize")
#cb_study.optimize(cb_objective, n_trials=100)
#print("Best score:", cb_study.best_value)
#print("Best params:", cb_study.best_params)
print('XGBOOST')
print()

model = XGBRegressor(**xgb_params)
xgb_oof_preds = np.zeros(len(X))
xgb_models, xgb_scores=[],[]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in kf.split(X, y):
        print('Fold:', len(xgb_models) + 1)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        xgb_oof_preds[val_idx] = model.predict(X_val)
        acc = mean_squared_error(y_val, model.predict(X_val), squared=False)
        xgb_scores.append(acc), xgb_models.append(model)
        print('Accuracy:', acc)
print('XGB ACCURACY: ', np.mean(xgb_scores))


def xgb_objective(trial):
    # Suggest weights for the ensemble
    learning_rate=trial.suggest_float("learning_rate", 0.001, 0.12, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 4)
    n_estimators = trial.suggest_int("n_estimators", 200, 1000)

    xgb=XGBRegressor(learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators,random_state=0,n_jobs=-1)
    X_train, X_val, y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=0)
    xgb.fit(X_train, y_train)
    preds=xgb.predict(X_val)
    score = mean_squared_error(y_val, preds, squared=False)
    return score

#xgb_study = optuna.create_study(direction="minimize")
#xgb_study.optimize(xgb_objective, n_trials=100)
#print("Best score:", xgb_study.best_value)
#print("Best params:", xgb_study.best_params)


print('LGBM')
print()
lgbm_model = LGBMRegressor(**lgbm_params)
lgbm_models, lgbm_scores=[],[]
lgbm_oof_preds = np.zeros(len(X))

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in kf.split(X, y):
        print('Fold:', len(lgbm_models) + 1)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        lgbm_model.fit(X_train, y_train)
        lgbm_oof_preds[val_idx]=lgbm_model.predict(X_val)
        acc = mean_squared_error(y_val, lgbm_model.predict(X_val), squared=False)
        lgbm_scores.append(acc), lgbm_models.append(lgbm_model)
        print('Accuracy:', acc)
print('LGBM ACCURACY: ', np.mean(lgbm_scores))
#lgbm_oof_preds = sum(lgbm_model.predict(X_val) for lgbm_model in lgbm_models) / len(lgbm_models)
y_val = y_val


def lgbm_objective(trial):
    # Suggest weights for the ensemble
    learning_rate=trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    n_estimators = trial.suggest_int("n_estimators", 200, 1000)

    lgbm=LGBMRegressor(learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators,random_state=0,n_jobs=-1)
    X_train, X_val, y_train,y_val=train_test_split(X,y,test_size=0.25,random_state=0)
    lgbm.fit(X_train, y_train)
    preds=lgbm.predict(X_val)
    score = mean_squared_error(y_val, preds, squared=False)
    return score

#lgbm_study = optuna.create_study(direction="minimize")
#lgbm_study.optimize(lgbm_objective, n_trials=100)
#print("Best score:", lgbm_study.best_value)
#print("Best params:", lgbm_study.best_params)


X_stack = pd.DataFrame({
    'xgb' : xgb_oof_preds,
    'lgbm' : lgbm_oof_preds,
    'cb' : cb_oof_preds
})
xgb_test_preds = sum(model.predict(test[Features]) for model in xgb_models) / len(xgb_models)
lgbm_test_preds = sum(lgbm_model.predict(test[Features]) for lgbm_model in lgbm_models) / len(lgbm_models)

cb_test_preds = sum(cb_model.predict(test[cat_features]) for cb_model in cb_models) / len(cb_models)


def objective(trial):
    # Suggest weights for the ensemble
    w_lgbm = trial.suggest_float("w_lgbm", 0, 1)
    w_cb = trial.suggest_float("w_cb", 0, 1-w_lgbm)
    w_xgb = 1 - w_lgbm - w_cb   # ensures sum to 1

    final_preds = (w_lgbm * X_stack['lgbm'] +w_xgb*X_stack['xgb'] + w_cb * X_stack['cb'] )
    score = mean_squared_error(y, final_preds, squared = False) 
    return score

#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=500)
#print("Best score:", study.best_value)
#print("Best params:", study.best_params)


w_lgbm = 0.5
w_cb = 0.25
w_xgb = 1-w_lgbm - w_cb
preds = w_lgbm * lgbm_test_preds + w_xgb * xgb_test_preds + w_cb * cb_test_preds
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': preds})
submission.to_csv('submission.csv', index=False)
display(submission.head())

