import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import combinations
import optuna

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


GPU = True
append_original = True


vis_feature_eng = False

try_interations = False

try_target_encode = False

include_ColorWaterproof = False

mean_enc_size = False
mean_enc_material = False
keep_weightcapbin = False
mean_enc_weightcapbin = True

tune_lgbm = False
tune_xgbm = False
tune_cat = False


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
original = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.drop('id', axis = 1, inplace = True)
test.drop('id', axis = 1, inplace = True)
original.drop('id', axis = 1, inplace = True)

if append_original:
    train = pd.concat([train, original])
    train = train.reset_index(drop = True)


conts = train.select_dtypes(include = np.number).columns.tolist()
conts.remove('Compartments')
cats = [col for col in train.columns if col not in conts]

conts.remove('Price')


for col in conts:
    train[col] = train[col].astype('float32')
    test[col] = test[col].astype('float32')

for col in cats:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


X = train.drop('Price', axis = 1).copy()
y = train['Price'].copy()

Xtest = test.copy()


print(X.shape)
print(len(y))


def cross_val(X, y, model):
    _X = X.copy()
    _y = y.copy()
    kfold = KFold(n_splits = 5)

    scores = []
    oof_preds = np.zeros(len(X))

    
    for i, (train_index, val_index) in enumerate(kfold.split(_X)):
        X_train = _X.iloc[train_index, :].copy()
        y_train = _y[train_index].copy()

        X_val= _X.iloc[val_index, :].copy()
        y_val = _y[val_index].copy()


        model.fit(X_train, y_train)
        fold_preds = model.predict(X_val)
        oof_preds[val_index] = fold_preds

        fold_score = mean_squared_error(y_val, fold_preds, squared=False)
        scores.append(fold_score)

    print(f"\n Mean CV Score: {np.mean(scores)}")

    return np.mean(scores), oof_preds



lgbm = lgb.LGBMRegressor(verbose = -1,
                        device = 'gpu' if GPU else 'cpu')      

cross_val(X, y, lgbm)


lgbm.fit(X, y)


lgb.plot_importance(lgbm)


X_ohe = pd.get_dummies(X, columns = cats)


lgbm_ohe = lgb.LGBMRegressor(verbose = -1)
lgbm_ohe.fit(X_ohe, y)


lgb.plot_importance(lgbm_ohe)


X_bins = X.copy()

bins_list = [4, 8, 12, 16, 20, 24, 30, 40, 60] 

if vis_feature_eng:
    for bins in bins_list:
        print(f"{bins} bins")
        X_bins['WeightCap Binned'] = pd.cut(X_bins['Weight Capacity (kg)'], bins=bins, labels = [f'bin_{i}' for i in range(bins)])
        lgbm = lgb.LGBMRegressor(verbose = -1)     
        cross_val(X_bins, y, lgbm)


bins = 30
X['WeightCap Binned'] = pd.cut(X['Weight Capacity (kg)'], bins=bins, labels = [f'bin_{i}' for i in range(bins)])
Xtest['WeightCap Binned'] = pd.cut(Xtest['Weight Capacity (kg)'], bins=bins, labels = [f'bin_{i}' for i in range(bins)])


cats.append('WeightCap Binned')


# don't mean encode first. target encode in the each CV fold.


def cross_val_mean_enc(X, y, model, encode_col):
    _X = X.copy()
    _y = y.copy()
    kfold = KFold(n_splits = 5)

    scores = []
    oof_preds = np.zeros(len(X))

    
    for i, (train_index, val_index) in enumerate(kfold.split(_X)):
        X_train = _X.iloc[train_index, :].copy()
        y_train = _y[train_index].copy()

        X_train_grouped = X_train.copy()
        X_train_grouped['Price'] = y_train
        X_train_grouped = X_train_grouped.groupby(encode_col, as_index = False)['Price'].mean()
        X_train_grouped = pd.DataFrame(X_train_grouped)
        X_train_grouped = X_train_grouped.rename(columns = {'Price' : 'mean_target_enc'})

        X_train = X_train.merge(X_train_grouped, on = encode_col, how = "left")

        X_val= _X.iloc[val_index, :].copy()
        X_val = X_val.merge(X_train_grouped, on = encode_col, how = "left")
        y_val = _y[val_index].copy()


        model.fit(X_train, y_train)
        fold_preds = model.predict(X_val)
        oof_preds[val_index] = fold_preds

        fold_score = mean_squared_error(y_val, fold_preds, squared=False)
        scores.append(fold_score)

    print(f"\n Mean CV Score: {np.mean(scores)}")

    return np.mean(scores), oof_preds



lgbm = lgb.LGBMRegressor(verbose = -1,
                        device = 'gpu' if GPU else 'cpu')      



if try_target_encode:
    for col in cats:
        print(f"Trying mean target encoding {col}")
        cross_val_mean_enc(X, y, lgbm, col)


def encode_X(X, Xtest, y, encode_col):
    _X = X.copy()
    _y = y.copy()
    _Xtest = Xtest.copy()
    
    X_grouped = _X.copy()
    X_grouped['Price'] = _y
    X_grouped = X_grouped.groupby(encode_col, as_index = False)['Price'].mean()
    X_grouped = pd.DataFrame(X_grouped)
    X_grouped = X_grouped.rename(columns = {'Price' : f'mean_target_enc_{encode_col}'})

    _X = _X.merge(X_grouped, on = encode_col, how = "left")
    _Xtest = _Xtest.merge(X_grouped, on = encode_col, how = "left")
    return _X, _Xtest

if mean_enc_size:
    X, Xtest = encode_X(X, Xtest, y,'Size')


if mean_enc_material:
    X, Xtest = encode_X(X, Xtest, y,'Material')


if mean_enc_weightcapbin:
    X, Xtest = encode_X(X, Xtest, y,'WeightCap Binned') 


if not keep_weightcapbin:
    X.drop('WeightCap Binned', axis = 1, inplace = True)
    Xtest.drop('WeightCap Binned', axis = 1, inplace = True)
    cats.remove('WeightCap Binned')


def convert_to_key(col1, col2):
    combined_and_sort = sorted([col1, col2])
    str_output = ''.join(combined_and_sort)
    return str(str_output)


if try_interations:


    
    pairs = {convert_to_key('col_1', 'col_2') : 0.001, convert_to_key('col_2', 'col_3') : 0.001}
    
    for col_i in cats:
    
        for col_j in cats:

            lgbm = lgb.LGBMRegressor(verbose = -1, device = 'gpu' if GPU else 'cpu') 
    
            X_Feat_eng = X.copy()
    
            if col_i != col_j and convert_to_key(col_i, col_j) not in pairs.keys():
                    print(f"New feature: {convert_to_key(col_i, col_j)}")
    
                    X_Feat_eng[convert_to_key(col_i, col_j)] = X_Feat_eng[col_i].astype(str) + "_" + X_Feat_eng[col_j].astype(str)
                    X_Feat_eng[convert_to_key(col_i, col_j)] = X_Feat_eng[convert_to_key(col_i, col_j)].astype('category')

                    try:
                        score, oof_preds = cross_val(X_Feat_eng, y, lgbm)
                    except:
                        lgbm = lgb.LGBMRegressor(verbose = -1, device = 'cpu') 
                        score, oof_preds = cross_val(X_Feat_eng, y, lgbm)
    
                    pairs[convert_to_key(col_i, col_j)] = score


if include_ColorWaterproof:

    X['ColorWaterproof'] = X['Color'].astype(str) + "_" + X['Waterproof'].astype(str)
    Xtest['ColorWaterproof'] = Xtest['Color'].astype(str) + "_" + Xtest['Waterproof'].astype(str)
    X['ColorWaterproof'] = X['ColorWaterproof'].astype('category')
    Xtest['ColorWaterproof'] = Xtest['ColorWaterproof'].astype('category')
    cats.append('ColorWaterproof')


Xcat = X.copy()
Xtestcat = Xtest.copy()

for col in cats:
    Xcat[col] = Xcat[col].astype(str)
    Xtestcat[col] = Xtestcat[col].astype(str)


def objective_lgbm(trial):
    params = {        
            "n_estimators": trial.suggest_int("n_estimators", 50, 2000, step=100),
            "max_depth":trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 7, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.),
            "num_leaves": trial.suggest_int("num_leaves", 2^4+1, 2^10+1),
    }   
    
    lgbm = lgb.LGBMRegressor(**params,
                                     verbose = -1, 
                                     device = 'gpu' if GPU else 'cpu')


    score, oof_preds = cross_val(X, y, lgbm)
    return score


if tune_lgbm:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_lgbm, n_trials=100)
    trial = study.best_trial


def objective_xgbm(trial):
    params = {        
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=100),
        "max_depth":trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.),
        "gamma": trial.suggest_float("gamma", 0.01, 1.0),
}    
    
    xgbm = xgb.XGBRegressor(**params,
                        enable_categorical = True,
                        tree_method = 'hist' if GPU else None,
                        verbosity = 0, 
                        objective = 'reg:squarederror',
                        device = 'cuda' if GPU else None)


    score, oof_preds = cross_val(X, y, xgbm)
    return score

if tune_xgbm:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_xgbm, n_trials=100)
    trial = study.best_trial


#params = {
#    'iterations': trial.suggest_int('iterations', 100, 1000), 
#    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),  
#    'depth': trial.suggest_int('depth', 1, 16),  
#    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),  
#    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
#}


def objective_cbm(trial):
    params = {
    'iterations': trial.suggest_int('iterations', 100, 2000), 
    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),  
    'depth': trial.suggest_int('depth', 1, 16),  
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),  
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
}  
    
    cbm = cb.CatBoostRegressor(**params,
                               cat_features = cats, verbose = False,
                          task_type="GPU" if GPU else "CPU")


    score, oof_preds = cross_val(Xcat, y, cbm)
    return score

if tune_cat:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_cbm, n_trials=20)
    trial = study.best_trial


tuned_params = {'n_estimators': 1150, 'max_depth': 7, 'min_child_weight': 14, 'learning_rate': 0.01923331287284719, 'subsample': 0.7934084377542973, 'colsample_bytree': 0.20164190306049304, 'reg_alpha': 0.9840478385725728, 'reg_lambda': 1.7963220335344459, 'num_leaves': 7}

lgbm = lgb.LGBMRegressor(**tuned_params,
                         verbose = -1, device = 'gpu' if GPU else 'cpu')
lgbm.fit(X,y)
lgbm_preds = lgbm.predict(Xtest)


lgb.plot_importance(lgbm)


tuned_params = {'n_estimators': 950, 'max_depth': 6, 'min_child_weight': 3, 'learning_rate': 0.009833085121752218, 'subsample': 0.9504678668570071, 'colsample_bytree': 0.1827797129259961, 'reg_alpha': 9.827832242145242, 'reg_lambda': 8.352519984345205, 'gamma': 0.13937610785116036}

xgbm = xgb.XGBRegressor(**tuned_params,
                        enable_categorical = True,
                        tree_method = 'hist' if GPU else None,
                        verbosity = 0, 
                        objective = 'reg:squarederror',
                        device = 'cuda' if GPU else None)
xgbm.fit(X,y)
xgbm_preds = xgbm.predict(Xtest)


tuned_params = {'iterations': 942, 'early_stopping_rounds': 34, 'depth': 6, 'l2_leaf_reg': 4.01802018825334, 'learning_rate': 0.015544786870961344}

cbm = cb.CatBoostRegressor(**tuned_params, 
                          cat_features = cats, verbose = False,
                          task_type="GPU" if GPU else "CPU")
cbm.fit(Xcat,y)
cbm_preds = cbm.predict(Xtestcat)


preds = 0.1 * lgbm_preds + 0.1 * xgbm_preds + 0.8 * cbm_preds


sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')

sub['Price'] = preds


sub.to_csv('submission.csv', index = False)


sub

