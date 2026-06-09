


!pip install uv -q
!uv pip install -q scikit-learn==1.6.1 xgboost==2.1.4 --system


import pandas as pd, numpy as np
from tqdm.notebook import tqdm
from gc import collect

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold as SKF
from xgboost import XGBClassifier as XGBC
from lightgbm import LGBMClassifier as LGBMC, log_evaluation, early_stopping
from catboost import CatBoostClassifier as CBC



target   = "IsDefault"
n_splits = 5


train  = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/train.csv", index_col = "Stock code")
test   = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/test.csv", index_col = "Stock code")
sub_fl = pd.DataFrame(index = test.index, columns = [target], dtype = np.int8).fillna(0)

Xtrain = train.drop(target, axis=1)
ytrain = train[target]
Xtest  = test.copy()

for df in [Xtrain, Xtest] :
    df.columns = [f"col{i}" for i in range(len(df.columns) )]

cat_cols = Xtrain.select_dtypes(["object", "string", "category"]).columns
Xtrain[cat_cols] = Xtrain[cat_cols].fillna("missing").astype("category")
Xtest[cat_cols]  = Xtest[cat_cols].fillna("missing").astype("category")

Xtrain.index = range(len(Xtrain))
ytrain.index = range(len(Xtrain))
Xtest.index  = range(len(Xtest))


print(f"\n ================== OFFLINE MODEL TRAINING ================== \n")

cv           = SKF(n_splits = n_splits, random_state = 42, shuffle = True)
oof_preds    = np.zeros(len(Xtrain))
mdl_preds    = np.zeros(len(Xtest))
pp_oof_preds = np.zeros(len(Xtrain))
pp_mdl_preds = np.zeros(len(Xtest))

for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(Xtrain, ytrain) ):
    print(f"\n ============== FOLD {fold_nb} ==============")

    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.loc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.loc[dev_idx]
    print(f"\n---> Shapes {Xtr.shape} {Xdev.shape} {ytr.shape} {ydev.shape}")

    model = \
    XGBC(
        n_estimators       = 300,
        random_state       = 42,
        objective          = "binary:logistic",
        eval_metric        = "logloss",
        reg_alpha          = 0.001,
        reg_lambda         = 0.001,
        colsample_bytree   = 0.35,
        colsample_bylevel  = 0.40,
        device             = "cpu",
        enable_categorical = True,    
    )

    model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], verbose = 50)
    dev_preds  = model.predict_proba(Xdev)[:,1]
    test_preds = model.predict_proba(Xtest)[:,1]
    score      = log_loss(ydev, dev_preds)
    score1     = f1_score(ydev, np.round(dev_preds))
    print(f"\n---> Log-loss = {score:,.8f} F1 raw = {score1 :,.8f}")
    
    oof_preds[dev_idx] = dev_preds
    mdl_preds = mdl_preds + (test_preds / n_splits)

    cutoffs = {}
    for cutoff in np.arange(0.05, 0.96, 0.005) :
        score = f1_score(ydev, np.where(dev_preds >= cutoff,1, 0))
        cutoffs[np.round(cutoff,4)] = score
        
    max_key   = max(cutoffs, key=cutoffs.get)
    max_value = cutoffs[max_key]
    print(
        f"---> F1 rounded = {max_value :,.8f} Best cutoff = {max_key :,.4f}"
    )

    pp_dev_preds          = np.where(dev_preds >= max_key, 1, 0)
    pp_oof_preds[dev_idx] = pp_dev_preds
    pp_mdl_preds          = pp_mdl_preds + (np.where(test_preds >= max_key, 1, 0) / n_splits)
    
score = f1_score(ytrain, np.round(oof_preds, 0))
print(f"\n---> Overall F1 score = {score :,.8f} | raw")

score = f1_score(ytrain, pp_oof_preds)
print(f"---> Overall F1 score = {score :,.8f} | rounded")


sub_fl[target] = np.uint8(mdl_preds.round())
sub_fl.to_csv("submission.csv", index = True)

!ls
print()
!head submission.csv

