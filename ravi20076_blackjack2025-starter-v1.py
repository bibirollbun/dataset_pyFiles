


!uv pip install -q xgboost==2.1.4 scikit-learn==1.6.1 lightgbm==4.6.0 --system

import numpy as np, pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import KFold as KF
from gc import collect
from tqdm.notebook import tqdm

from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping
from catboost import CatBoostRegressor as CBR


target   = "ev"
n_splits = 20
version_nb = "V1_3"


train  = pd.read_csv(f"/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv", index_col = "id")
test   = pd.read_csv(f"/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv", index_col = "id")
sub_fl = pd.read_csv(f"/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv", index_col = "id")

print(f"---> Shapes = {train.shape} {test.shape}")


OOF_Preds = {}
Mdl_Preds = {}
cv        = KF(n_splits = n_splits, random_state = 42, shuffle  = True)


print()

oof_preds   = np.zeros(len(train))
test_preds  = 0

print(f"\n ================== XGBOOST ================== \n")

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(train, train[target]) ):

    print(
        f"\n ============== FOLD {fold_nb + 1} ============== \n"
    )
    
    Xtr, ytr   = train.loc[train_idx].drop(target, axis = 1), train.loc[train_idx, target]
    Xdev, ydev = train.loc[dev_idx].drop(target, axis = 1),   train.loc[dev_idx, target]

    model = \
    XGBR(
        n_estimators     = 25_000,
        random_state     = 42,
        learning_rate    = 0.02,
        max_depth        = 3,
        colsample_bytree = 0.55,
        subsample        = 0.65,
        objective        = "reg:squarederror",
        eval_metric      = "rmse", 
        early_stopping_rounds = 100,
        enable_categorical = True,
        verbosity = 0,
    )
        
    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)],
        verbose  = 2500,
    )

    dev_preds  = model.predict(Xdev)
    preds      = model.predict(test.copy())
    score      = mean_squared_error(ydev, dev_preds)

    oof_preds[dev_idx]     = dev_preds
    test_preds             = test_preds +  ( preds / n_splits )
    print(f"\n---> Score = {score :,.8f}")


OOF_Preds["XGB1R"] = oof_preds
Mdl_Preds["XGB1R"] = test_preds

score = mean_squared_error(train[target], oof_preds) * 10**8
print(f"\n\n---> Final OOF score scaled up = {score :,.4f}\n\n")


print()

oof_preds   = np.zeros(len(train))
test_preds  = 0

print(f"\n ================== LIGHTGBM ================== \n")

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(train, train[target]) ):

    print(
        f"\n ============== FOLD {fold_nb + 1} ============== \n"
    )
    
    Xtr, ytr   = train.loc[train_idx].drop(target, axis = 1), train.loc[train_idx, target]
    Xdev, ydev = train.loc[dev_idx].drop(target, axis = 1),   train.loc[dev_idx, target]

    model = \
    LGBMR(
        n_estimators     = 25_000,
        random_state     = 42,
        learning_rate    = 0.02,
        max_depth        = 3,
        colsample_bytree = 0.55,
        subsample        = 0.65,
        objective        = "regression_l2",
        metric           = "rmse", 
        verbosity        = -1,        
    )
        
    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)],
        callbacks = [log_evaluation(2500), early_stopping(100, verbose = False)]
    )

    dev_preds  = model.predict(Xdev)
    preds      = model.predict(test.copy())
    score      = mean_squared_error(ydev, dev_preds)

    oof_preds[dev_idx]     = dev_preds
    test_preds             = test_preds +  ( preds / n_splits )
    print(f"\n---> Score = {score :,.8f}")


OOF_Preds["LGBM1R"] = oof_preds
Mdl_Preds["LGBM1R"] = test_preds

score = mean_squared_error(train[target], oof_preds) * 10**8
print(f"\n\n---> Final OOF score scaled up = {score :,.4f}\n\n")


print()

oof_preds   = np.zeros(len(train))
test_preds  = 0

print(f"\n ================== CATBOOST ================== \n")

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(train, train[target]) ):

    print(
        f"\n ============== FOLD {fold_nb + 1} ============== \n"
    )
    
    Xtr, ytr   = train.loc[train_idx].drop(target, axis = 1), train.loc[train_idx, target]
    Xdev, ydev = train.loc[dev_idx].drop(target, axis = 1),   train.loc[dev_idx, target]

    model = \
    CBR(
        iterations    = 20_000,
        loss_function = "RMSE",
        eval_metric   = "RMSE",
        max_depth     = 3,
        l2_leaf_reg   = 0.35,
        learning_rate = 0.02,
        early_stopping_rounds = 100,
        verbose       = 2500,
    )
        
    model.fit(
        Xtr, ytr, 
        eval_set = [(Xdev, ydev)],
        verbose  = 2500,
    )

    dev_preds  = model.predict(Xdev)
    preds      = model.predict(test.copy())
    score      = mean_squared_error(ydev, dev_preds)

    oof_preds[dev_idx]     = dev_preds
    test_preds             = test_preds +  ( preds / n_splits )
    print(f"\n---> Score = {score :,.8f}")


OOF_Preds["CB1R"] = oof_preds
Mdl_Preds["CB1R"] = test_preds

score = mean_squared_error(train[target], oof_preds) * 10**8
print(f"\n\n---> Final OOF score scaled up = {score :,.4f}\n\n")


%%time 

score = \
mean_squared_error( 
    train[target], 
    np.mean( pd.DataFrame(OOF_Preds).values, axis=1)
) * 10**8
print(f"---> Final OOF score scaled up = {score :,.4f}")

sub_fl[target] = np.mean( pd.DataFrame(Mdl_Preds).values, axis=1)
sub_fl.to_csv("submission.csv")

!ls
print()
!head submission.csv

