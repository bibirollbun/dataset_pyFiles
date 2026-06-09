import pandas as pd 
import numpy as np
from sklearn.ensemble import IsolationForest 
from pygam import LinearGAM, s 

pd.set_option('display.max_columns', None) 

# read in data
train = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/train.csv')
test = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/test.csv') 

# only two strikes 
train = train[train['strikes'] == 2]

# drop unnecessary columns
# I prefer to be pitch type-agnostic, this might be hurting me but I can live with that 
train.drop(columns=['pitch_name','pitch_type','is_strike','strikes'], inplace=True) 
test.drop(columns=['pitch_name','pitch_type','strikes'], inplace=True) 

def basic_features(df): 
    df['inning_topbot'] = df['inning_topbot'].map({'Top':1,'Bot':0}) 
    
    for col in ['on_3b','on_2b','on_1b']: 
        df[col] = df[col].astype(int) 
    
    for col in ['pfx_x','release_pos_x']: 
        df[col] = np.where(df['p_throws'] == 'L', df[col].mul(-1), df[col]) 

    for col in ['stand','p_throws']: 
        df[col] = df[col].map({'R':1,'L':0})

    return df 

train = basic_features(train) 
test = basic_features(test)

# isolation forest to use 
iforest = IsolationForest(n_estimators = 100, max_samples = 'auto') 

# how unique is a pitch w/r/t movement, arm angle, and velo 
def calculate_outlier_scores(df): 
    iforest.fit_predict(df[['pfx_x','pfx_z','arm_angle','release_speed']]) 
    score = iforest.decision_function(df[['pfx_x','pfx_z','arm_angle','release_speed']]) 
    df['outlier_score'] = score
        
    return df 

train = calculate_outlier_scores(train) 
test = calculate_outlier_scores(test) 

# calculate 'unexpectedness' of movement based on arm angle using GAM 
def arm_angle_gam(df): 
    for col in ['pfx_x','pfx_z']: 
        X = df[['release_speed','arm_angle']]
        y = df[col] 
        gam = LinearGAM(s(0) + s(1)).fit(X, y) 
        df[col + '_resid'] = df[col] - gam.predict(X) 
        
    return df 

train = arm_angle_gam(train)
test = arm_angle_gam(test)

# split training and testing data into swings and takes 
train_swing = train[train['bat_speed'].isna() == False] 
test_swing = test[test['bat_speed'].isna() == False] 
train_take = train[train['bat_speed'].isna() == True] 
test_take = test[test['bat_speed'].isna() == True] 

from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
import xgboost as xgb 
import optuna 
 
# let's work on swings first 
X_swing = train_swing.drop(columns=['index','k'], axis=1)  
y_swing = train_swing['k'] 

def xgb_objective_swing(trial): 
    params = {
        'objective':'binary:logistic', 
        'n_estimators': 200, 
        'verbosity': 0, 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    } 
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=711) 
    cv_scores = [] 
    
    for train_idx, val_idx in cv.split(X_swing, y_swing): 
        X_swing_train, X_swing_val = X_swing.iloc[train_idx], X_swing.iloc[val_idx] 
        y_swing_train, y_swing_val = y_swing.iloc[train_idx], y_swing.iloc[val_idx]  
    
        model = xgb.XGBClassifier(**params, random_seed=711)  
        model.fit(X_swing_train, y_swing_train, verbose=0) 
        val_preds = model.predict_proba(X_swing_val)[:, 1] 
        ll_score = log_loss(y_swing_val, val_preds) 
        cv_scores.append(ll_score)
    
    return np.mean(cv_scores)

swing_study = optuna.create_study(direction='minimize') 
swing_study.optimize(xgb_objective_swing, n_trials=50)

best_swing_params = swing_study.best_params 

full_swing_model = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    n_estimators = 200, 
    learning_rate = best_swing_params['learning_rate'], 
    max_depth = best_swing_params['max_depth'], 
    subsample = best_swing_params['subsample'], 
    colsample_bytree = best_swing_params['colsample_bytree'], 
    min_child_weight = best_swing_params['min_child_weight']
    )
full_swing_model.fit(X_swing, y_swing) 

swing_importance = {X_swing.columns[i]: full_swing_model.feature_importances_[i] 
                    for i in range(len(X_swing.columns))} 

# moving onto takes 
# bat speed and swing length are no longer features 
X_take = train_take.drop(columns=['index','k','bat_speed','swing_length'], axis=1) 
y_take = train_take['k'] 

def xgb_objective_take(trial): 
    params = {
        'objective':'binary:logistic', 
        'n_estimators': 200, 
        'verbosity': 0, 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    } 
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=711) 
    cv_scores = [] 
    
    for train_idx, val_idx in cv.split(X_take, y_take): 
        X_take_train, X_take_val = X_take.iloc[train_idx], X_take.iloc[val_idx] 
        y_take_train, y_take_val = y_take.iloc[train_idx], y_take.iloc[val_idx]  
    
        model = xgb.XGBClassifier(**params, random_seed=711)  
        model.fit(X_take_train, y_take_train, verbose=0) 
        val_preds = model.predict_proba(X_take_val)[:, 1] 
        ll_score = log_loss(y_take_val, val_preds) 
        cv_scores.append(ll_score)
    
    return np.mean(cv_scores)

take_study = optuna.create_study(direction='minimize') 
take_study.optimize(xgb_objective_take, n_trials=50) 

best_take_params = take_study.best_params

full_take_model = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    n_estimators = 200, 
    learning_rate = best_take_params['learning_rate'], 
    max_depth = best_take_params['max_depth'], 
    subsample = best_take_params['subsample'], 
    colsample_bytree = best_take_params['colsample_bytree'], 
    min_child_weight = best_take_params['min_child_weight']
    )
full_take_model.fit(X_take, y_take)

take_importance = {X_take.columns[i]: full_take_model.feature_importances_[i] 
                    for i in range(len(X_take.columns))} 


# preeictions on swings in test data
test_swing['k'] = full_swing_model.predict_proba(
    test_swing.drop(columns='index', axis=1))[:, 1] 

# predictions on take in test data 
test_take['k'] = full_take_model.predict_proba(
    test_take.drop(columns=['index','bat_speed','swing_length'], axis=1))[:, 1]


# aggregrate results, then sort by index to ensure correct submission 
all_test = pd.concat([test_swing, test_take]) 
all_test = all_test.sort_index() 

sample_sol = pd.read_csv('/Users/justinchoi/Downloads/nwds-k/sample_solution.csv') 
sample_sol['k'] = all_test['k'] 
sample_sol.to_csv('arm_angle_submission_v9.csv', index=False)

